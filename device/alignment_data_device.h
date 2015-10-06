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

#include "../types.h"
#include "../alignment_data.h"

namespace firepony {

template <target_system system>
struct alignment_header_device : public alignment_header_storage<system>
{
    void download(const alignment_header_host& host)
    {
        this->chromosome_lengths.copy(host.chromosome_lengths);
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

        const persistent_allocation<system, uint16> chromosome;
        const persistent_allocation<system, uint32> alignment_start;
        const persistent_allocation<system, uint32> alignment_stop;
        const persistent_allocation<system, uint32> mate_chromosome;
        const persistent_allocation<system, uint32> mate_alignment_start;
        const persistent_allocation<system, int32> inferred_insert_size;
        const persistent_allocation<system, cigar_op> cigars;
        const persistent_allocation<system, uint32> cigar_start;
        const persistent_allocation<system, uint32> cigar_len;
        typename packed_vector<system, 4>::const_view reads;
        const persistent_allocation<system, uint32> read_start;
        const persistent_allocation<system, uint32> read_len;
        const persistent_allocation<system, uint8> qualities;
        const persistent_allocation<system, uint32> qual_start;
        const persistent_allocation<system, uint32> qual_len;
        const persistent_allocation<system, uint16> flags;
        const persistent_allocation<system, uint8> mapq;
        const persistent_allocation<system, uint32> read_group;

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
            this->chromosome.copy(host.chromosome);
        } else {
            this->chromosome.clear();
        }

        if (this->data_mask & AlignmentDataMask::ALIGNMENT_START)
        {
            this->alignment_start.copy(host.alignment_start);
        } else {
            this->alignment_start.clear();
        }

        if (this->data_mask & AlignmentDataMask::ALIGNMENT_STOP)
        {
            this->alignment_stop.copy(host.alignment_stop);
        } else {
            this->alignment_stop.clear();
        }

        if (this->data_mask & AlignmentDataMask::MATE_CHROMOSOME)
        {
            this->mate_chromosome.copy(host.mate_chromosome);
        } else {
            this->mate_chromosome.clear();
        }

        if (this->data_mask & AlignmentDataMask::MATE_ALIGNMENT_START)
        {
            this->mate_alignment_start.copy(host.mate_alignment_start);
        } else {
            this->mate_alignment_start.clear();
        }

        if (this->data_mask & AlignmentDataMask::INFERRED_INSERT_SIZE)
        {
            this->inferred_insert_size.copy(host.inferred_insert_size);
        } else {
            this->inferred_insert_size.clear();
        }

        if (this->data_mask & AlignmentDataMask::CIGAR)
        {
            this->cigars.copy(host.cigars);
            this->cigar_start.copy(host.cigar_start);
            this->cigar_len.copy(host.cigar_len);
        } else {
            this->cigars.clear();
            this->cigar_start.clear();
            this->cigar_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::READS)
        {
            this->reads.copy(host.reads);
            this->read_start.copy(host.read_start);
            this->read_len.copy(host.read_len);
        } else {
            this->reads.clear();
            this->read_start.clear();
            this->read_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::QUALITIES)
        {
            this->qualities.copy(host.qualities);
            this->qual_start.copy(host.qual_start);
            this->qual_len.copy(host.qual_len);
        } else {
            this->qualities.clear();
            this->qual_start.clear();
            this->qual_len.clear();
        }

        if (this->data_mask & AlignmentDataMask::FLAGS)
        {
            this->flags.copy(host.flags);
        } else {
            this->flags.clear();
        }

        if (this->data_mask & AlignmentDataMask::MAPQ)
        {
            this->mapq.copy(host.mapq);
        } else {
            this->mapq.clear();
        }

        if (this->data_mask & AlignmentDataMask::READ_GROUP)
        {
            this->read_group.copy(host.read_group);
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

