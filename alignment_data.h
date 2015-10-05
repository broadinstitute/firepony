/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector>
#include <map>

#include "types.h"
#include "string_database.h"
#include "sequence_database.h"

namespace firepony {

// contains information referenced by an alignment data batch
// xxxnsubtil: a better name might be in order
template <target_system system>
struct alignment_header_storage
{
    // the length of each chromosome in the reference
    vector<system, uint32> chromosome_lengths;

    struct const_view
    {
        typename vector<system, uint32>::const_view chromosome_lengths;
    };

    operator const_view() const
    {
        const_view v = {
            chromosome_lengths,
        };

        return v;
    }
};

struct alignment_header_host : public alignment_header_storage<host>
{
    string_database read_groups_db;
};

enum AlignmentFlags
{
    // the read is paired in sequencing, no matter whether it is mapped in a pair
    PAIRED = 1,
    // the read is mapped in a proper pair
    PROPER_PAIR = 2,
    // the read itself is unmapped; conflicts with PROPER_PAIR
    UNMAP = 4,
    // the mate is unmapped
    MATE_UNMAP = 8,
    // the read is mapped to the reverse strand
    REVERSE = 16,
    // the mate is mapped to the reverse strand
    MATE_REVERSE = 32,
    // this is read1
    READ1 = 64,
    // this is read2
    READ2 = 128,
    // not primary alignment
    SECONDARY = 256,
    // QC failure
    QC_FAIL = 512,
    // optical or PCR duplicate
    DUPLICATE = 1024,
    // supplementary alignment
    SUPPLEMENTARY = 2048,
};

namespace AlignmentDataMask
{
    enum
    {
        NAME                 = 0x0001,
        CHROMOSOME           = 0x0002,
        ALIGNMENT_START      = 0x0004,
        ALIGNMENT_STOP       = 0x0008,
        MATE_CHROMOSOME      = 0x0010,
        MATE_ALIGNMENT_START = 0x0020,
        INFERRED_INSERT_SIZE = 0x0040,
        CIGAR                = 0x0080,
        READS                = 0x0100,
        QUALITIES            = 0x0200,
        FLAGS                = 0x0400,
        MAPQ                 = 0x0800,

        // list of tags that we require
        READ_GROUP           = 0x1000,
    };
}

// CRQ: cigars, reads, qualities
struct CRQ_index
{
    const uint32 cigar_start, cigar_len;
    const uint32 read_start, read_len;
    const uint32 qual_start, qual_len;

    CUDA_HOST_DEVICE CRQ_index(const uint32 cigar_start, const uint32 cigar_len,
                               const uint32 read_start, const uint32 read_len,
                               const uint32 qual_start, const uint32 qual_len)
        : cigar_start(cigar_start),
          cigar_len(cigar_len),
          read_start(read_start),
          read_len(read_len),
          qual_start(qual_start),
          qual_len(qual_len)
    { }
};

template <target_system system>
struct alignment_batch_storage
{
    uint32 num_reads;
    uint32 max_read_size;
    uint32 data_mask;

    // chromosome index of the read
    vector<system, uint16> chromosome;
    // the reference position of the first base in the read
    vector<system, uint32> alignment_start;
    // (1-based and inclusive) alignment stop position
    vector<system, uint32> alignment_stop;
    // integer representation of the mate's chromosome
    vector<system, uint32> mate_chromosome;
    // (1-based and inclusive) mate's alignment start position
    vector<system, uint32> mate_alignment_start;
    // inferred insert size
    vector<system, int32> inferred_insert_size;

    // cigar ops
    vector<system, cigar_op> cigars;
    // cigar index vectors
    vector<system, uint32> cigar_start;
    vector<system, uint32> cigar_len;

    // read data (4 bits per base pair)
    packed_vector<system, 4> reads;
    // read index vectors
    vector<system, uint32> read_start;
    vector<system, uint32> read_len;

    // quality data
    vector<system, uint8> qualities;
    // quality index vectors
    vector<system, uint32> qual_start;
    vector<system, uint32> qual_len;

    // alignment flags
    vector<system, uint16> flags;
    // mapping qualities
    vector<system, uint8> mapq;

    // read group ID
    vector<system, uint32> read_group;

    // prevent storage creation on the device
    CUDA_HOST alignment_batch_storage() { }
};

struct alignment_batch_host : public alignment_batch_storage<host>
{
    // data that never gets copied to the device
    std::vector<std::string> name;          // read name
    resident_segment_map chromosome_map;    // map of chromosomes referenced by this batch

    const CRQ_index crq_index(uint32 read_id) const
    {
        return CRQ_index(cigar_start[read_id],
                         cigar_len[read_id],
                         read_start[read_id],
                         read_len[read_id],
                         qual_start[read_id],
                         qual_len[read_id]);
    }

    void reset(uint32 data_mask, uint32 batch_size, sequence_database_host& reference)
    {
        num_reads = 0;
        max_read_size = 0;
        this->data_mask = data_mask;

        name.clear();
        chromosome.clear();
        alignment_start.clear();
        alignment_stop.clear();
        mate_chromosome.clear();
        mate_alignment_start.clear();
        inferred_insert_size.clear();

        cigars.clear();
        cigar_start.clear();
        cigar_len.clear();

        reads.clear();
        read_start.clear();
        read_len.clear();

        qualities.clear();
        qual_start.clear();
        qual_len.clear();

        flags.clear();
        mapq.clear();

        read_group.clear();

        if (data_mask & AlignmentDataMask::NAME)
        {
            name.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            chromosome.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            alignment_start.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            alignment_stop.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            mate_chromosome.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            mate_alignment_start.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            inferred_insert_size.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            cigars.reserve(batch_size * 32);
            cigar_start.reserve(batch_size);
            cigar_len.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            reads.reserve(batch_size * 100);
            read_start.reserve(batch_size);
            read_len.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            qualities.reserve(batch_size * 100);
            qual_start.reserve(batch_size);
            qual_len.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            flags.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            mapq.reserve(batch_size);
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            read_group.reserve(batch_size);
        }

        chromosome_map = reference.empty_segment_map();
    }
};

} // namespace firepony

