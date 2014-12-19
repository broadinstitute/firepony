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

#include "../types.h"
#include "../alignment_data.h"
#include "primitives/cuda.h"

namespace firepony {

template <target_system system>
struct alignment_header_device : public alignment_header_storage<system>
{
    void download(const alignment_header_host& host)
    {
        this->chromosome_lengths = host.chromosome_lengths;
    }
};

template <target_system system>
struct alignment_header
{
    // the host-side data comes from outside impl, so it's a reference
    const alignment_header_host& host;
    alignment_header_device<system> device;

    alignment_header(const alignment_header_host& host)
        : host(host)
    { }

    void download(void)
    {
        device.download(host);
    }
};

template <target_system system>
struct alignment_batch_device : public alignment_batch_storage<system>
{
    typedef alignment_batch_storage<system> base;

    CUDA_DEVICE const CRQ_index crq_index(uint32 read_id) const
    {
        return CRQ_index(this->cigar_start[read_id],
                         this->cigar_len[read_id],
                         this->read_start[read_id],
                         this->read_len[read_id],
                         this->qual_start[read_id],
                         this->qual_len[read_id]);
    }

    struct const_view
    {
        uint32 num_reads;
        uint32 max_read_size;

        typename vector<system, uint32>::const_view chromosome;
        typename vector<system, uint32>::const_view alignment_start;
        typename vector<system, uint32>::const_view alignment_stop;
        typename vector<system, uint32>::const_view mate_chromosome;
        typename vector<system, uint32>::const_view mate_alignment_start;
        typename vector<system, int16>::const_view inferred_insert_size;
        typename vector<system, cigar_op>::const_view cigars;
        typename vector<system, uint32>::const_view cigar_start;
        typename vector<system, uint32>::const_view cigar_len;
        typename packed_vector<system, 4>::const_view reads;
        typename vector<system, uint32>::const_view read_start;
        typename vector<system, uint32>::const_view read_len;
        typename vector<system, uint8>::const_view qualities;
        typename vector<system, uint32>::const_view qual_start;
        typename vector<system, uint32>::const_view qual_len;
        typename vector<system, uint16>::const_view flags;
        typename vector<system, uint8>::const_view mapq;
        typename vector<system, uint32>::const_view read_group;

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
                base::num_reads,
                base::max_read_size,

                base::chromosome,
                base::alignment_start,
                base::alignment_stop,
                base::mate_chromosome,
                base::mate_alignment_start,
                base::inferred_insert_size,
                base::cigars,
                base::cigar_start,
                base::cigar_len,
                base::reads,
                base::read_start,
                base::read_len,
                base::qualities,
                base::qual_start,
                base::qual_len,
                base::flags,
                base::mapq,
                base::read_group,
        };

        return v;
    }

    void download(const alignment_batch_host& host)
    {
        this->num_reads = host.num_reads;
        this->max_read_size = host.max_read_size;
        this->data_mask = host.data_mask;

        if (this->data_mask & AlignmentDataMask::CHROMOSOME)
        {
            this->chromosome = host.chromosome;
        } else {
            this->chromosome.clear();
        }

        if (this->data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            this->alignment_start = host.alignment_start;
        } else {
            this->alignment_start.clear();
        }

        if (this->data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            this->alignment_stop = host.alignment_stop;
        } else {
            this->alignment_stop.clear();
        }

        if (this->data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            this->mate_chromosome = host.mate_chromosome;
        } else {
            this->mate_chromosome.clear();
        }

        if (this->data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            this->mate_alignment_start = host.mate_alignment_start;
        } else {
            this->mate_alignment_start.clear();
        }

        if (this->data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            this->inferred_insert_size = host.inferred_insert_size;
        } else {
            this->inferred_insert_size.clear();
        }

        if (this->data_mask & AlignmentDataMask::CIGAR)
        {
            this->cigars = host.cigars;
            this->cigar_start = host.cigar_start;
            this->cigar_len = host.cigar_len;
        } else {
            this->cigars.clear();
            this->cigar_start.clear();
            this->cigar_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::READS)
        {
            this->reads = host.reads;
            this->read_start = host.read_start;
            this->read_len = host.read_len;
        } else {
            this->reads.clear();
            this->read_start.clear();
            this->read_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::QUALITIES)
        {
            this->qualities = host.qualities;
            this->qual_start = host.qual_start;
            this->qual_len = host.qual_len;
        } else {
            this->qualities.clear();
            this->qual_start.clear();
            this->qual_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::FLAGS)
        {
            this->flags = host.flags;
        } else {
            this->flags.clear();
        }

        if (this->data_mask & AlignmentDataMask::MAPQ)
        {
            this->mapq = host.mapq;
        } else {
            this->mapq.clear();
        }

        if (this->data_mask & AlignmentDataMask::READ_GROUP)
        {
            this->read_group = host.read_group;
        } else {
            this->read_group.clear();
        }
    }
};

template <target_system system>
struct alignment_batch
{
    // host data comes from outside impl and can change
    const alignment_batch_host *host;
    alignment_batch_device<system> device;

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

