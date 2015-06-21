/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "../sequence_database.h"
#include "../string_database.h"

#include "reference.h"
#include "../command_line.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

#include "../mmap.h"
#include "../serialization.h"

namespace firepony {

#include <thrust/iterator/transform_iterator.h>

struct iupac16 : public thrust::unary_function<char, uint8>
{
    uint8 operator() (char in)
    {
        return from_nvbio::char_to_iupac16(in);
    }
};

reference_file_handle::reference_file_handle(const std::string filename, uint32 consumers)
  : filename(filename), consumers(consumers)
{
    sequence_mutexes.resize(consumers);
}

void reference_file_handle::consumer_lock(const uint32 consumer_id)
{
    sequence_mutexes[consumer_id].lock();
}

void reference_file_handle::consumer_unlock(const uint32 consumer_id)
{
    sequence_mutexes[consumer_id].unlock();
}

bool reference_file_handle::load_index()
{
    std::string index_fname;
    std::ifstream index_fstream;

    // check if we have an index
    index_fname = filename + ".fai";
    index_fstream.open(index_fname);
    if (index_fstream.fail())
    {
        index_available = false;
        return false;
    }

    std::string line;
    while(std::getline(index_fstream, line))
    {
        // faidx format: <sequence name> <sequence read_len> <file offset> <line blen> <line len>
        // fields are separated by \t

        // replace all \t with spaces to ease parsing
        std::replace(line.begin(), line.end(), '\t', ' ');

        std::stringstream ss(line);
        std::string name;
        size_t len;
        size_t offset;

        ss >> name;
        ss >> len;
        ss >> offset;

        reference_index[string_database::hash(name)] = offset;
    }

    index_available = true;
    return true;
}

reference_file_handle *reference_file_handle::open(const std::string filename, uint32 consumers, bool try_mmap)
{
    reference_file_handle *handle = new reference_file_handle(filename, consumers);

    handle->file_handle.open(filename);
    if (handle->file_handle.fail())
    {
        fprintf(stderr, "error opening %s\n", filename.c_str());
        delete handle;
        return nullptr;
    }

    if (!handle->load_index())
    {
        // no index present, load entire reference
        fprintf(stderr, "WARNING: index not available for reference file %s, loading entire reference\n", filename.c_str());
        while (handle->load_next_sequence())
            ;
    } else {
        fprintf(stderr, "loaded index for %s\n", filename.c_str());

        if (try_mmap)
        {
            shared_memory_file shmem;
            bool ret;

            ret = shared_memory_file::open(&shmem, filename.c_str());
            if (ret == true)
            {
                serialization::unserialize(&handle->sequence_data, shmem.data);
                shmem.unmap();
            }
        }
    }

    return handle;
}

void reference_file_handle::producer_lock(void)
{
    for(uint32 id = 0; id < sequence_mutexes.size(); id++)
        consumer_lock(id);
}

void reference_file_handle::producer_unlock(void)
{
    for(uint32 id = 0; id < sequence_mutexes.size(); id++)
        consumer_unlock(id);
}

bool reference_file_handle::make_sequence_available(const std::string& sequence_name)
{
    // check if we already loaded the requested sequence
    if (sequence_data.sequence_names.lookup(sequence_name) != uint32(-1))
    {
        // nothing to do
        return true;
    }

    if (!index_available)
    {
        // if the index is not available, we can't load sequences on demand
        return false;
    }

    // search for the sequence name in the reference file index
    auto it = reference_index.find(string_database::hash(sequence_name));
    if (it == reference_index.end())
    {
        // sequence not found in index, can't load
        return false;
    }

    // found it, grab the offset
    size_t offset = it->second;

    // we must seek backwards to the beginning of the header
    // (faidx points at the beginning of the sequence data, but we need to parse the header)
    char c;
    do {
        file_handle.seekg(offset);
        file_handle >> c;
        file_handle.seekg(offset);

        if (c != '>')
            offset--;
    } while(c != '>');

    producer_lock();
    bool ret = load_next_sequence();
    producer_unlock();

    if (ret == false)
    {
        fprintf(stderr, "error loading sequence %s\n", sequence_name.c_str());
    } else {
        fprintf(stderr, "+");
    }

    fflush(stderr);

    return ret;
}

// filter carriage returns from a string
static void filter_cr(std::string& string)
{
    if (string[string.size() - 1] == '\r')
    {
        string.pop_back();
    }
}

bool reference_file_handle::load_next_sequence(void)
{
    sequence_storage<host> *h;
    uint32 seq_id;

    if (file_handle.fail())
    {
        return false;
    }

    std::string sequence_name;
    if (!std::getline(file_handle, sequence_name))
    {
        return false;
    }

    if (sequence_name[0] != '>')
    {
        // we only accept fasta files
        return false;
    }

    // remove carriage returns
    filter_cr(sequence_name);
    // remove the header delimiter
    sequence_name.erase(0, 1);
    // tokenize
    std::replace(sequence_name.begin(), sequence_name.end(), ' ', '\0');

    // allocate a sequence
    seq_id = sequence_data.sequence_names.insert(sequence_name);
    h = sequence_data.new_entry(seq_id);

    std::vector<char> sequence(16 * 1024 * 1024);
    size_t seq_ptr = 0;

    // read until we get another sequence header or we reach the end of the file
    std::string line;
    while(std::getline(file_handle, line))
    {
        if (line[0] == '>')
        {
            // we read the header for the next sequence
            // rewind back to the start and break
            file_handle.seekg(-(line.size() + 1), std::ios_base::cur);
            break;
        }

        // remove carriage returns
        filter_cr(line);

        // make sure we have enough memory to store the sequence
        if (seq_ptr + line.size() >= sequence.size())
        {
            sequence.resize(sequence.size() * 2);
        }

        // ... and store it
        memcpy(&sequence[seq_ptr], &line[0], line.size());
        seq_ptr += line.size();
    }

    // apply base pair conversions
    for(size_t i = 0; i < sequence.size(); i++)
    {
        // convert lower-case base pairs to upper-case
        if (sequence[i] == 'a')
        {
            sequence[i] = 'A';
        }

        if (sequence[i] == 'c')
        {
            sequence[i] = 'C';
        }

        if (sequence[i] == 'g')
        {
            sequence[i] = 'G';
        }

        if (sequence[i] == 't')
        {
            sequence[i] = 'T';
        }

        // convert everything besides ACGT to N
        if (sequence[i] != 'A' &&
            sequence[i] != 'C' &&
            sequence[i] != 'G' &&
            sequence[i] != 'T')
        {
            sequence[i] = 'N';
        }
    }

    // encode and store the sequence in the output
    h->bases.resize(seq_ptr);
    assign(seq_ptr,
           thrust::make_transform_iterator(sequence.begin(), iupac16()),
           h->bases.stream_at_index(0));

    return h->bases.size() > 0;
}

} // namespace firepony
