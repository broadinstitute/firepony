/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#include <gamgee/fastq.h>
#include <gamgee/fastq_iterator.h>
#include <gamgee/fastq_reader.h>

#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "../sequence_data.h"
#include "../string_database.h"

#include "reference.h"
#include "../command_line.h"

#include "../device/util.h"
#include "../device/from_nvbio/dna.h"

namespace firepony {

#include <thrust/iterator/transform_iterator.h>

struct iupac16 : public thrust::unary_function<char, uint8>
{
    uint8 operator() (char in)
    {
        return from_nvbio::char_to_iupac16(in);
    }
};

static bool load_record(sequence_data_host *output, const gamgee::Fastq& record, uint32 data_mask)
{
    auto& h = *output;
    h.data_mask = data_mask;

    h.num_sequences++;

    if (data_mask & SequenceDataMask::BASES)
    {
        std::string sequence = record.sequence();

        const size_t seq_start = h.bases.size();
        const size_t seq_len = sequence.size();

        h.sequence_bp_start.push_back(seq_start);
        h.sequence_bp_len.push_back(seq_len);

        h.bases.resize(seq_start + seq_len);

        assign(sequence.size(),
               thrust::make_transform_iterator(sequence.begin(), iupac16()),
               h.bases.stream_at_index(seq_start));
    }

    if (data_mask & SequenceDataMask::QUALITIES)
    {
        assert(!"unimplemented");
        return false;
    }

    if (data_mask & SequenceDataMask::NAMES)
    {
        uint32 seq_id = output->sequence_names.insert(record.name());
        h.sequence_id.push_back(seq_id);
    }

    return true;
}

// loader for sequence data
static bool load_reference(sequence_data_host *output, const char *filename, uint32 data_mask)
{
    for (gamgee::Fastq& record : gamgee::FastqReader(std::string(filename)))
    {
        bool ret = load_record(output, record, data_mask);
        if (ret == false)
            return false;
    }

    return true;
}

static bool load_one_sequence(sequence_data_host *output, const std::string filename, size_t file_offset, uint32 data_mask)
{
    // note: we can't reuse the existing ifstream as gamgee for some reason wants to take ownership of the pointer and destroy it
    std::ifstream *file_stream = new std::ifstream();
    file_stream->open(filename);
    file_stream->seekg(file_offset);

    gamgee::FastqReader reader(file_stream);
    gamgee::Fastq record = *(reader.begin());

    return load_record(output, record, data_mask);
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
        // faidx format: <sequence name> <sequence len> <file offset> <line blen> <line len>
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

reference_file_handle *reference_file_handle::open(const std::string filename, uint32 data_mask)
{
    reference_file_handle *handle = new reference_file_handle(filename, data_mask);

    if (!handle->load_index())
    {
        // no index present, load entire reference
        fprintf(stderr, "WARNING: index not available for reference file %s, loading entire reference\n", filename.c_str());
        load_reference(&handle->sequence_data, filename.c_str(), data_mask);
    } else {
        fprintf(stderr, "loaded index for %s\n", filename.c_str());

        handle->file_handle.open(filename);
        if (handle->file_handle.fail())
        {
            fprintf(stderr, "error opening %s\n", filename.c_str());
            delete handle;
            return nullptr;
        }
    }

    return handle;
}

bool reference_file_handle::make_sequence_available(const std::string& sequence_name)
{
    if (!index_available)
    {
        return (sequence_data.sequence_names.lookup(sequence_name) != uint32(-1));
    } else {
        if (sequence_data.sequence_names.lookup(sequence_name) == uint32(-1))
        {

            auto it = reference_index.find(string_database::hash(sequence_name));
            if (it == reference_index.end())
            {
                fprintf(stderr, "ERROR: sequence %s not found in reference index\n", sequence_name.c_str());
                return false;
            }

            fprintf(stderr, "loading reference sequence %s...", sequence_name.c_str());
            fflush(stderr);
            size_t offset = it->second;

            // we must seek backwards to the beginning of the header
            // (faidx points at the beginning of the sequence data, but gamgee needs to parse the header)
            char c;
            do {
                file_handle.seekg(offset);
                file_handle >> c;
                file_handle.seekg(offset);

                if (c != '>')
                    offset--;
            } while(c != '>');

            bool ret = load_one_sequence(&sequence_data, filename, offset, data_mask);
            if (ret == true)
            {
                fprintf(stderr, " done\n");
            }

            return ret;
        } else {
            return true;
        }
    }
}

} // namespace firepony
