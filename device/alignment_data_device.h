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

#include "../alignment_data.h"
#include "device_types.h"
#include "primitives/cuda.h"

namespace firepony {

struct alignment_header_device : public alignment_header_storage<target_system_tag>
{
    void download(const alignment_header_host& host)
    {
        chromosome_lengths = host.chromosome_lengths;
    }
};

struct alignment_header
{
    // the host-side data comes from outside impl, so it's a reference
    const alignment_header_host& host;
    alignment_header_device device;

    alignment_header(const alignment_header_host& host)
        : host(host)
    { }

    void download(void)
    {
        device.download(host);
    }
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
        uint32 max_read_size;

        D_Vector<uint32>::const_view chromosome;
        D_Vector<uint32>::const_view alignment_start;
        D_Vector<uint32>::const_view alignment_stop;
        D_Vector<uint32>::const_view mate_chromosome;
        D_Vector<uint32>::const_view mate_alignment_start;
        D_Vector<int16>::const_view inferred_insert_size;
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
                max_read_size,

                chromosome,
                alignment_start,
                alignment_stop,
                mate_chromosome,
                mate_alignment_start,
                inferred_insert_size,
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

    void download(const alignment_batch_host& host)
    {
        num_reads = host.num_reads;
        max_read_size = host.max_read_size;
        data_mask = host.data_mask;

        if (data_mask & AlignmentDataMask::CHROMOSOME)
        {
            chromosome = host.chromosome;
        } else {
            chromosome.clear();
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            alignment_start = host.alignment_start;
        } else {
            alignment_start.clear();
        }

        if (data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            alignment_stop = host.alignment_stop;
        } else {
            alignment_stop.clear();
        }

        if (data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            mate_chromosome = host.mate_chromosome;
        } else {
            mate_chromosome.clear();
        }

        if (data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            mate_alignment_start = host.mate_alignment_start;
        } else {
            mate_alignment_start.clear();
        }

        if (data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            inferred_insert_size = host.inferred_insert_size;
        } else {
            inferred_insert_size.clear();
        }

        if (data_mask & AlignmentDataMask::CIGAR)
        {
            cigars = host.cigars;
            cigar_start = host.cigar_start;
            cigar_len = host.cigar_len;
        } else {
            cigars.clear();
            cigar_start.clear();
            cigar_len.clear();
        }

        if (data_mask & AlignmentDataMask::READS)
        {
            reads = host.reads;
            read_start = host.read_start;
            read_len = host.read_len;
        } else {
            reads.clear();
            read_start.clear();
            read_len.clear();
        }

        if (data_mask & AlignmentDataMask::QUALITIES)
        {
            qualities = host.qualities;
            qual_start = host.qual_start;
            qual_len = host.qual_len;
        } else {
            qualities.clear();
            qual_start.clear();
            qual_len.clear();
        }

        if (data_mask & AlignmentDataMask::FLAGS)
        {
            flags = host.flags;
        } else {
            flags.clear();
        }

        if (data_mask & AlignmentDataMask::MAPQ)
        {
            mapq = host.mapq;
        } else {
            mapq.clear();
        }

        if (data_mask & AlignmentDataMask::READ_GROUP)
        {
            read_group = host.read_group;
        } else {
            read_group.clear();
        }
    }
};

struct alignment_batch
{
    // host data comes from outside impl and can change
    const alignment_batch_host *host;
    alignment_batch_device device;

    alignment_batch()
        : host(NULL)
    { }

    void download(const alignment_batch_host *host)
    {
        this->host = host;
        device.download(*host);
    }
};

} // namespace firepony

