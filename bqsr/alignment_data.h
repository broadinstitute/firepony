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

#include <gamgee/sam.h>

#include "bqsr_types.h"
#include "util.h"

#include "primitives/cuda.h"

// contains information referenced by an alignment data batch
// xxxnsubtil: a better name might be in order
struct alignment_header
{
    // the length of each chromosome in the reference
    // xxxnsubtil: this is an ugly hack
    H_Vector<uint32> chromosome_lengths;
    D_Vector<uint32> d_chromosome_lengths;

    string_database read_groups_db;

    struct const_view
    {
        bqsr::vector<target_system_tag, uint32>::const_view chromosome_lengths;
    };

    operator const_view() const
    {
        const_view v = {
                d_chromosome_lengths,
        };

        return v;
    }
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
        NAME                 = 0x001,
        CHROMOSOME           = 0x002,
        ALIGNMENT_START      = 0x004,
        ALIGNMENT_STOP       = 0x008,
        MATE_CHROMOSOME      = 0x010,
        MATE_ALIGNMENT_START = 0x020,
        CIGAR                = 0x040,
        READS                = 0x080,
        QUALITIES            = 0x100,
        FLAGS                = 0x200,
        MAPQ                 = 0x400,

        // list of tags that we require
        READ_GROUP           = 0x800,
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
    template <typename T> using Vector = bqsr::vector<system_tag, T>;
    template <uint32 bits> using PackedVector = bqsr::packed_vector<system_tag, bits>;

    uint32 num_reads;

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

struct alignment_batch_device : public alignment_batch_storage<target_system_tag>
{
    CUDA_DEVICE const CRQ_index crq_index(uint32 read_id) const
    {
        return CRQ_index(cigar_start[read_id],
                         cigar_len[read_id],
                         read_start[read_id],
                         read_len[read_id],
                         qual_start[read_id],
                         qual_len[read_id]);
    }

    struct const_view
    {
        uint32 num_reads;

        D_Vector<uint32>::const_view chromosome;
        D_Vector<uint32>::const_view alignment_start;
        D_Vector<uint32>::const_view alignment_stop;
        D_Vector<uint32>::const_view mate_chromosome;
        D_Vector<uint32>::const_view mate_alignment_start;
        D_Vector<cigar_op>::const_view cigars;
        D_Vector<uint32>::const_view cigar_start;
        D_Vector<uint32>::const_view cigar_len;
        D_PackedVector<4>::const_view reads;
        D_Vector<uint32>::const_view read_start;
        D_Vector<uint32>::const_view read_len;
        D_Vector<uint8>::const_view qualities;
        D_Vector<uint32>::const_view qual_start;
        D_Vector<uint32>::const_view qual_len;
        D_Vector<uint16>::const_view flags;
        D_Vector<uint8>::const_view mapq;
        D_Vector<uint32>::const_view read_group;

        CUDA_HOST_DEVICE const CRQ_index crq_index(uint32 read_id) const
        {
            return CRQ_index(cigar_start[read_id],
                             cigar_len[read_id],
                             read_start[read_id],
                             read_len[read_id],
                             qual_start[read_id],
                             qual_len[read_id]);
        }
    };

    operator const_view() const
    {
        const_view v = {
                num_reads,

                chromosome,
                alignment_start,
                alignment_stop,
                mate_chromosome,
                mate_alignment_start,
                cigars,
                cigar_start,
                cigar_len,
                reads,
                read_start,
                read_len,
                qualities,
                qual_start,
                qual_len,
                flags,
                mapq,
                read_group,
        };

        return v;
    }
};

struct alignment_batch_host : public alignment_batch_storage<host_tag>
{
    // data that never gets copied to the device
    H_Vector<std::string> name;

    CUDA_HOST const CRQ_index crq_index(uint32 read_id) const
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

        name.clear();
        chromosome.clear();
        alignment_start.clear();
        alignment_stop.clear();
        mate_chromosome.clear();
        mate_alignment_start.clear();

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

struct alignment_batch
{
    uint32 data_mask;

    alignment_batch_host host;
    alignment_batch_device device;

    void reset(uint32 mask, uint32 batch_size)
    {
        data_mask = mask;
        host.reset(data_mask, batch_size);
    }

    void download(void)
    {
        device.num_reads = host.num_reads;

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            device.chromosome = host.chromosome;
        } else {
            device.chromosome.clear();
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            device.alignment_start = host.alignment_start;
        } else {
            device.alignment_start.clear();
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            device.alignment_stop = host.alignment_stop;
        } else {
            device.alignment_stop.clear();
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            device.mate_chromosome = host.mate_chromosome;
        } else {
            device.mate_chromosome.clear();
        }

        if (data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            device.mate_alignment_start = host.mate_alignment_start;
        } else {
            device.mate_alignment_start.clear();
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            device.cigars = host.cigars;
            device.cigar_start = host.cigar_start;
            device.cigar_len = host.cigar_len;
        } else {
            device.cigars.clear();
            device.cigar_start.clear();
            device.cigar_len.clear();
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            device.reads = host.reads;
            device.read_start = host.read_start;
            device.read_len = host.read_len;
        } else {
            device.reads.clear();
            device.read_start.clear();
            device.read_len.clear();
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            device.qualities = host.qualities;
            device.qual_start = host.qual_start;
            device.qual_len = host.qual_len;
        } else {
            device.qualities.clear();
            device.qual_start.clear();
            device.qual_len.clear();
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            device.flags = host.flags;
        } else {
            device.flags.clear();
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            device.mapq = host.mapq;
        } else {
            device.mapq.clear();
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            device.read_group = host.read_group;
        } else {
            device.read_group.clear();
        }
    }
};
