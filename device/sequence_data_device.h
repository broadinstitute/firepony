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

struct sequence_data_device : public sequence_data_storage<target_system_tag>
{
    size_t download(const sequence_data_host& host)
    {
        size_t num_bytes = 0;
#define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.view.f[0]) * f.size();
#define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * f.m_storage.size();

        num_sequences = host.view.num_sequences;
        data_mask = host.view.data_mask;

        if (data_mask & SequenceDataMask::BASES)
        {
            bases.copy_from_view(host.view.bases);
            sequence_bp_start.copy_from_view(host.view.sequence_bp_start);
            sequence_bp_len.copy_from_view(host.view.sequence_bp_len);

            TRACK_PACKED_VECTOR_SIZE(bases);
            TRACK_VECTOR_SIZE(sequence_bp_start);
            TRACK_VECTOR_SIZE(sequence_bp_len);
        }

        if (data_mask & SequenceDataMask::QUALITIES)
        {
            qualities.copy_from_view(host.view.qualities);
            sequence_qual_start.copy_from_view(host.view.sequence_qual_start);
            sequence_qual_len.copy_from_view(host.view.sequence_qual_len);

            TRACK_VECTOR_SIZE(qualities);
            TRACK_VECTOR_SIZE(sequence_qual_start);
            TRACK_VECTOR_SIZE(sequence_qual_len);
        }

        if (data_mask & SequenceDataMask::NAMES)
        {
            sequence_id.copy_from_view(host.view.sequence_id);
            TRACK_VECTOR_SIZE(sequence_id);
        }

        return num_bytes;
    #undef TRACK_VECTOR_SIZE
    #undef TRACK_PACKED_VECTOR_SIZE
    }

};

struct sequence_data
{
    const sequence_data_host& host;
    sequence_data_device device;

    sequence_data(const sequence_data_host& host)
        : host(host)
    { }

    size_t download(void)
    {
        return device.download(host);
    }
};

} // namespace firepony
