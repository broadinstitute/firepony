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
        : num_sequences(0)
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

struct sequence_data_host
{
    string_database sequence_names;

    // the const_view that wraps all sequence data
    // this will be populated either from malloc-backed storage or from a memory-mapped file
    sequence_data_storage<host>::const_view view;

    // the containers that store the actual data
    // only one of these is used
    sequence_data_storage<host> host_malloc_container;
    shared_memory_file host_mmap_container;

    size_t serialized_size(void)
    {
        size_t ret = 0;

        ret += serialization::serialized_size(host_malloc_container.data_mask);
        ret += sequence_names.serialized_size();
        ret += serialization::serialized_size(host_malloc_container.num_sequences);
        ret += serialization::serialized_size(host_malloc_container.bases.m_size);
        ret += serialization::serialized_size(host_malloc_container.bases.m_storage);
        ret += serialization::serialized_size(host_malloc_container.qualities);
        ret += serialization::serialized_size(host_malloc_container.sequence_id);
        ret += serialization::serialized_size(host_malloc_container.sequence_bp_start);
        ret += serialization::serialized_size(host_malloc_container.sequence_bp_len);
        ret += serialization::serialized_size(host_malloc_container.sequence_qual_start);
        ret += serialization::serialized_size(host_malloc_container.sequence_qual_len);

        return ret;
    }

    void *serialize(void *out)
    {
        out = serialization::encode(out, &host_malloc_container.data_mask);
        out = sequence_names.serialize(out);
        out = serialization::encode(out, &host_malloc_container.num_sequences);

        // xxxnsubtil: fix packed_vector serialization at some point
        out = serialization::encode(out, &host_malloc_container.bases.m_size);
        out = serialization::encode(out, &host_malloc_container.bases.m_storage);

        out = serialization::encode(out, &host_malloc_container.qualities);
        out = serialization::encode(out, &host_malloc_container.sequence_id);
        out = serialization::encode(out, &host_malloc_container.sequence_bp_start);
        out = serialization::encode(out, &host_malloc_container.sequence_bp_len);
        out = serialization::encode(out, &host_malloc_container.sequence_qual_start);
        out = serialization::encode(out, &host_malloc_container.sequence_qual_len);

        return out;
    }

    void unserialize(shared_memory_file& shm)
    {
        void *in = shm.data;
        uint64 temp;

        in = serialization::decode(&view.data_mask, in);
        in = sequence_names.unserialize(in);
        in = serialization::decode(&view.num_sequences, in);

        uint32 m_size;
        in = serialization::decode(&m_size, in);
        in = serialization::decode(&temp, in);

        view.bases = firepony::packed_vector<host, 4>::const_view(in, m_size);
        in = static_cast<uint32*>(in) + divide_ri(m_size, firepony::packed_vector<host, 4>::SYMBOLS_PER_WORD);

        in = serialization::unwrap_vector_view(view.qualities, in);
        in = serialization::unwrap_vector_view(view.sequence_id, in);
        in = serialization::unwrap_vector_view(view.sequence_bp_start, in);
        in = serialization::unwrap_vector_view(view.sequence_bp_len, in);
        in = serialization::unwrap_vector_view(view.sequence_qual_start, in);
        in = serialization::unwrap_vector_view(view.sequence_qual_len, in);

        host_mmap_container = shm;
    }
};

} // namespace firepony
