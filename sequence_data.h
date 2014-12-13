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

#include "types.h"
#include "string_database.h"

namespace firepony {

namespace SequenceDataMask
{
    enum
    {
        BASES       = 0x001,
        QUALITIES   = 0x002,
        NAMES       = 0x004,
    };
}

template <target_system system>
struct sequence_data_storage
{
    // the generation counter is used to check if the GPU vs CPU versions are out of date
    uint32 generation;

    uint32 data_mask;
    uint32 num_sequences;

    packed_vector<system, 4> bases;
    vector<system, uint8> qualities;

    vector<system, uint32> sequence_id;
    // note: bases and quality indexes may not match if sequences are padded to dword length
    vector<system, uint64> sequence_bp_start;
    vector<system, uint64> sequence_bp_len;
    vector<system, uint64> sequence_qual_start;
    vector<system, uint64> sequence_qual_len;

    CUDA_HOST sequence_data_storage()
        : generation(0),
          num_sequences(0)
    { }

    struct const_view
    {
        uint32 data_mask;
        uint32 num_sequences;

        typename packed_vector<system, 4>::const_view bases;
        typename vector<system, uint8>::const_view qualities;
        typename vector<system, uint32>::const_view sequence_id;
        typename vector<system, uint64>::const_view sequence_bp_start;
        typename vector<system, uint64>::const_view sequence_bp_len;
        typename vector<system, uint64>::const_view sequence_qual_start;
        typename vector<system, uint64>::const_view sequence_qual_len;
    };

    CUDA_HOST operator const_view() const
    {
        const_view v = {
                data_mask,
                num_sequences,
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

struct sequence_data_host : public sequence_data_storage<host>
{
    string_database sequence_names;
};

} // namespace firepony
