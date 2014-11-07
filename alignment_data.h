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

#pragma once

#include <vector>
#include <map>

#include "types.h"
#include "device/primitives/vector.h"
#include "device/string_database.h"

namespace firepony {

// contains information referenced by an alignment data batch
// xxxnsubtil: a better name might be in order
template <typename system_tag>
struct alignment_header_storage
{
    template <typename T> using Vector = firepony::vector<system_tag, T>;

    // the length of each chromosome in the reference
    Vector<uint32> chromosome_lengths;

    struct const_view
    {
        vector<target_system_tag, uint32>::const_view chromosome_lengths;
    };

    operator const_view() const
    {
        const_view v = {
            chromosome_lengths,
        };

        return v;
    }
};

struct alignment_header_host : public alignment_header_storage<host_tag>
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
    const uint32& cigar_start, cigar_len;
    const uint32& read_start, read_len;
    const uint32& qual_start, qual_len;

    CUDA_HOST_DEVICE CRQ_index(const uint32& cigar_start, const uint32& cigar_len,
                               const uint32& read_start, const uint32& read_len,
                               const uint32& qual_start, const uint32& qual_len)
        : cigar_start(cigar_start),
          cigar_len(cigar_len),
          read_start(read_start),
          read_len(read_len),
          qual_start(qual_start),
          qual_len(qual_len)
    { }
};

template <typename system_tag>
struct alignment_batch_storage
{
    template <typename T> using Vector = vector<system_tag, T>;
    template <uint32 bits> using PackedVector = packed_vector<system_tag, bits>;

    uint32 num_reads;
    uint32 max_read_size;
    uint32 data_mask;

    // chromosome index of the read
    Vector<uint32> chromosome;
    // the reference position of the first base in the read
    Vector<uint32> alignment_start;
    // (1-based and inclusive) alignment stop position
    Vector<uint32> alignment_stop;
    // integer representation of the mate's chromosome
    Vector<uint32> mate_chromosome;
    // (1-based and inclusive) mate's alignment start position
    Vector<uint32> mate_alignment_start;
    // inferred insert size
    Vector<int16> inferred_insert_size;

    // cigar ops
    Vector<cigar_op> cigars;
    // cigar index vectors
    Vector<uint32> cigar_start;
    Vector<uint32> cigar_len;

    // read data (4 bits per base pair)
    PackedVector<4> reads;
    // read index vectors
    Vector<uint32> read_start;
    Vector<uint32> read_len;

    // quality data
    Vector<uint8> qualities;
    // quality index vectors
    Vector<uint32> qual_start;
    Vector<uint32> qual_len;

    // alignment flags
    Vector<uint16> flags;
    // mapping qualities
    Vector<uint8> mapq;

    // read group ID
    Vector<uint32> read_group;

    // prevent storage creation on the device
    CUDA_HOST alignment_batch_storage() { }
};

struct alignment_batch_host : public alignment_batch_storage<host_tag>
{
    // data that never gets copied to the device
    H_Vector<std::string> name;

    const CRQ_index crq_index(uint32 read_id) const
    {
        return CRQ_index(cigar_start[read_id],
                         cigar_len[read_id],
                         read_start[read_id],
                         read_len[read_id],
                         qual_start[read_id],
                         qual_len[read_id]);
    }

    void reset(uint32 data_mask, uint32 batch_size)
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
    }
};

} // namespace firepony

