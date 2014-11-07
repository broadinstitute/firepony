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
#include "mmap.h"

#include "primitives/cuda.h"

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

template <typename system_tag>
struct sequence_data_storage
{
    template <typename T> using Vector = vector<system_tag, T>;
    template <uint32 bits> using PackedVector = packed_vector<system_tag, bits>;

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

    struct const_view
    {
        uint32 num_sequences;

        typename PackedVector<4>::const_view bases;
        typename Vector<uint8>::const_view qualities;
        typename Vector<uint32>::const_view sequence_id;
        typename Vector<uint64>::const_view sequence_bp_start;
        typename Vector<uint64>::const_view sequence_bp_len;
        typename Vector<uint64>::const_view sequence_qual_start;
        typename Vector<uint64>::const_view sequence_qual_len;
    };

    CUDA_HOST operator const_view() const
    {
        const_view v = {
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

typedef sequence_data_storage<host_tag> sequence_data_host;
typedef sequence_data_storage<target_system_tag> sequence_data_device;

struct sequence_data
{
    uint32 data_mask;
    string_database sequence_names;

    // the const_view that wraps all sequence data
    // this will be populated either from malloc-backed storage or from a memory-mapped file
    sequence_data_host::const_view host;

    // the containers that store the actual data
    // only one of these is used
    sequence_data_host host_malloc_container;
    shared_memory_file host_mmap_container;

    sequence_data_device device;

    size_t serialized_size(void);
    void *serialize(void *out);
    void unserialize(shared_memory_file& shm);

    size_t download(void);
};

} // namespace firepony
