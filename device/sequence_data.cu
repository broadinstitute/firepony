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

#include "serialization.h"
#include "sequence_data.h"

namespace firepony {

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

    host.bases = packed_vector<host_tag, 4>::const_view(in, m_size);
    in = static_cast<uint32*>(in) + divide_ri(m_size, packed_vector<host_tag, 4>::SYMBOLS_PER_WORD);

    in = serialization::unwrap_vector_view(host.qualities, in);
    in = serialization::unwrap_vector_view(host.sequence_id, in);
    in = serialization::unwrap_vector_view(host.sequence_bp_start, in);
    in = serialization::unwrap_vector_view(host.sequence_bp_len, in);
    in = serialization::unwrap_vector_view(host.sequence_qual_start, in);
    in = serialization::unwrap_vector_view(host.sequence_qual_len, in);

    host_mmap_container = shm;
}

size_t sequence_data::download(void)
{
    size_t num_bytes = 0;
#define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.f[0]) * device.f.size();
#define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * device.f.m_storage.size();

    device.num_sequences = host.num_sequences;

    if (data_mask & SequenceDataMask::BASES)
    {
        device.bases.copy_from_view(host.bases);
        device.sequence_bp_start.copy_from_view(host.sequence_bp_start);
        device.sequence_bp_len.copy_from_view(host.sequence_bp_len);

        TRACK_PACKED_VECTOR_SIZE(bases);
        TRACK_VECTOR_SIZE(sequence_bp_start);
        TRACK_VECTOR_SIZE(sequence_bp_len);
    }

    if (data_mask & SequenceDataMask::QUALITIES)
    {
        device.qualities.copy_from_view(host.qualities);
        device.sequence_qual_start.copy_from_view(host.sequence_qual_start);
        device.sequence_qual_len.copy_from_view(host.sequence_qual_len);

        TRACK_VECTOR_SIZE(qualities);
        TRACK_VECTOR_SIZE(sequence_qual_start);
        TRACK_VECTOR_SIZE(sequence_qual_len);
    }

    if (data_mask & SequenceDataMask::NAMES)
    {
        device.sequence_id.copy_from_view(host.sequence_id);
        TRACK_VECTOR_SIZE(sequence_id);
    }

    return num_bytes;
#undef TRACK_VECTOR_SIZE
#undef TRACK_PACKED_VECTOR_SIZE
}

} // namespace firepony
