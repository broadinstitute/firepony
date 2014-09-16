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

#include "sequence_data.h"

size_t sequence_data::serialized_size(void)
{
    size_t ret = 0;

    ret += serialization::serialized_size(data_mask);
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

void *sequence_data::serialize(void *out)
{
    out = serialization::encode(out, &data_mask);
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

template <typename T>
static void *unwrap_view(T& view, void *in)
{
    uint64 size;

    in = serialization::decode(&size, in);

    view = T(typename T::iterator(in), typename T::size_type(size));
    in = (void *) (static_cast<typename T::pointer>(in) + size);

    return in;
}

void sequence_data::unserialize(shared_memory_file& shm)
{
    void *in = shm.data;
    uint64 temp;

    in = serialization::decode(&data_mask, in);
    in = sequence_names.unserialize(in);
    in = serialization::decode(&host.num_sequences, in);

    uint32 m_size;
    in = serialization::decode(&m_size, in);
    in = serialization::decode(&temp, in);

    host.bases = bqsr::packed_vector<host_tag, 4>::const_view(in, m_size);
    in = static_cast<uint32*>(in) + divide_ri(m_size, bqsr::packed_vector<host_tag, 4>::SYMBOLS_PER_WORD);

    in = unwrap_view(host.qualities, in);
    in = unwrap_view(host.sequence_id, in);
    in = unwrap_view(host.sequence_bp_start, in);
    in = unwrap_view(host.sequence_bp_len, in);
    in = unwrap_view(host.sequence_qual_start, in);
    in = unwrap_view(host.sequence_qual_len, in);

    host_mmap_container = shm;
}

void sequence_data::download(void)
{
    uint64 num_bytes = 0;

    device.num_sequences = host.num_sequences;

    if (data_mask & SequenceDataMask::BASES)
    {
        device.bases.copy_from_view(host.bases);
        device.sequence_bp_start.copy_from_view(host.sequence_bp_start);
        device.sequence_bp_len.copy_from_view(host.sequence_bp_len);
        num_bytes += host.bases.size() / 2 + host.sequence_bp_start.size() * 8 + host.sequence_bp_len.size() * 8;
    }

    if (data_mask & SequenceDataMask::QUALITIES)
    {
        device.qualities.copy_from_view(host.qualities);
        device.sequence_qual_start.copy_from_view(host.sequence_qual_start);
        device.sequence_qual_len.copy_from_view(host.sequence_qual_len);

        num_bytes += host.qualities.size() + host.sequence_qual_start.size() * 8 + host.sequence_qual_len.size() * 8;
    }

    if (data_mask & SequenceDataMask::NAMES)
    {
        device.sequence_id.copy_from_view(host.sequence_id);

        num_bytes += host.sequence_id.size() * 4;
    }

    printf("downloaded %lu MB of sequence data\n", num_bytes / (1024 * 1024));
}
