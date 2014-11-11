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
#define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.view.f[0]) * this->f.size();
#define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * this->f.m_storage.size();

        this->num_sequences = host.view.num_sequences;
        this->data_mask = host.view.data_mask;

        if (this->data_mask & SequenceDataMask::BASES)
        {
            this->bases.copy_from_view(host.view.bases);
            this->sequence_bp_start.copy_from_view(host.view.sequence_bp_start);
            this->sequence_bp_len.copy_from_view(host.view.sequence_bp_len);

            TRACK_PACKED_VECTOR_SIZE(bases);
            TRACK_VECTOR_SIZE(sequence_bp_start);
            TRACK_VECTOR_SIZE(sequence_bp_len);
        }

        if (this->data_mask & SequenceDataMask::QUALITIES)
        {
            this->qualities.copy_from_view(host.view.qualities);
            this->sequence_qual_start.copy_from_view(host.view.sequence_qual_start);
            this->sequence_qual_len.copy_from_view(host.view.sequence_qual_len);

            TRACK_VECTOR_SIZE(qualities);
            TRACK_VECTOR_SIZE(sequence_qual_start);
            TRACK_VECTOR_SIZE(sequence_qual_len);
        }

        if (this->data_mask & SequenceDataMask::NAMES)
        {
            this->sequence_id.copy_from_view(host.view.sequence_id);
            TRACK_VECTOR_SIZE(sequence_id);
        }

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
