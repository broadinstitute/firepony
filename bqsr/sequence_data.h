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

#include "bqsr_types.h"
#include "string_database.h"

#include "primitives/cuda.h"

namespace SequenceDataMask
{
    enum
    {
        BASES       = 0x001,
        QUALITIES   = 0x002,
        NAMES       = 0x004,
    };
}

template <typename system_tag>
struct sequence_data_storage
{
    template <typename T> using Vector = bqsr::vector<system_tag, T>;
    template <uint32 bits> using PackedVector = bqsr::packed_vector<system_tag, bits>;

    uint32 num_sequences;

    PackedVector<4> bases;
    Vector<uint8> qualities;

    Vector<uint32> sequence_id;
    // note: bases and quality indexes may not match if sequences are padded to dword length
    Vector<uint64> sequence_bp_start;
    Vector<uint64> sequence_bp_len;
    Vector<uint64> sequence_qual_start;
    Vector<uint64> sequence_qual_len;

    CUDA_HOST sequence_data_storage()
        : num_sequences(0)
    { }
};

struct sequence_data_device : public sequence_data_storage<target_system_tag>
{
    struct const_view
    {
        D_VectorDNA16::const_view bases;
        D_VectorU8::const_view qualities;
        D_VectorU32::const_view sequence_id;
        D_VectorU64::const_view sequence_bp_start;
        D_VectorU64::const_view sequence_bp_len;
        D_VectorU64::const_view sequence_qual_start;
        D_VectorU64::const_view sequence_qual_len;
    };

    operator const_view() const
    {
        const_view v = {
                bases,
                qualities,
                sequence_id,
                sequence_bp_start,
                sequence_bp_len,
                sequence_qual_start,
                sequence_qual_len,
        };

        return v;
    }
};

struct sequence_data_host : public sequence_data_storage<host_tag>
{
    string_database sequence_names;
};

struct sequence_data
{
    uint32 data_mask;

    sequence_data_host host;
    sequence_data_device device;

    void download(void)
    {
        uint64 num_bytes = 0;

        device.num_sequences = host.num_sequences;

        if (data_mask & SequenceDataMask::BASES)
        {
            device.bases = host.bases;
            device.sequence_bp_start = host.sequence_bp_start;
            device.sequence_bp_len = host.sequence_bp_len;

            num_bytes += host.bases.size() / 2 + host.sequence_bp_start.size() * 8 + host.sequence_bp_len.size() * 8;
        }

        if (data_mask & SequenceDataMask::QUALITIES)
        {
            device.qualities = host.qualities;
            device.sequence_qual_start = host.sequence_qual_start;
            device.sequence_qual_len = host.sequence_qual_len;

            num_bytes += host.qualities.size() + host.sequence_qual_start.size() * 8 + host.sequence_qual_len.size() * 8;
        }

        if (data_mask & SequenceDataMask::NAMES)
        {
            device.sequence_id = host.sequence_id;

            num_bytes += host.sequence_id.size() * 4;
        }

        printf("downloaded %lu MB of sequence data\n", num_bytes / (1024 * 1024));
    }
};
