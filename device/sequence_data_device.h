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

#pragma once

#include "../sequence_data.h"
#include "device_types.h"
#include "primitives/cuda.h"

namespace firepony {

template <target_system system>
struct sequence_data_device : public sequence_data_storage<system>
{
    size_t download(const sequence_data_host& host)
    {
        size_t num_bytes = 0;
#define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.f[0]) * this->f.size();
#define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * this->f.m_storage.size();

        this->num_sequences = host.num_sequences;
        this->data_mask = host.data_mask;

        if (this->data_mask & SequenceDataMask::BASES)
        {
            this->bases = host.bases;
            this->sequence_bp_start = host.sequence_bp_start;
            this->sequence_bp_len = host.sequence_bp_len;

            TRACK_PACKED_VECTOR_SIZE(bases);
            TRACK_VECTOR_SIZE(sequence_bp_start);
            TRACK_VECTOR_SIZE(sequence_bp_len);
        }

        if (this->data_mask & SequenceDataMask::QUALITIES)
        {
            this->qualities = host.qualities;
            this->sequence_qual_start = host.sequence_qual_start;
            this->sequence_qual_len = host.sequence_qual_len;

            TRACK_VECTOR_SIZE(qualities);
            TRACK_VECTOR_SIZE(sequence_qual_start);
            TRACK_VECTOR_SIZE(sequence_qual_len);
        }

        if (this->data_mask & SequenceDataMask::NAMES)
        {
            this->sequence_id = host.sequence_id;
            TRACK_VECTOR_SIZE(sequence_id);
        }

        this->generation = host.generation;

        return num_bytes;
    #undef TRACK_VECTOR_SIZE
    #undef TRACK_PACKED_VECTOR_SIZE
    }

};

template <target_system system>
struct sequence_data
{
    const sequence_data_host& host;
    sequence_data_device<system> device;

    sequence_data(const sequence_data_host& host)
        : host(host)
    { }

    size_t download(void)
    {
        return device.download(host);
    }
};

} // namespace firepony
