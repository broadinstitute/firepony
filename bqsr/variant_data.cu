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
#include "variant_data.h"

size_t variant_database::serialized_size(void)
{
    size_t ret = 0;

    ret += serialization::serialized_size(data_mask);
    ret += id_db.serialized_size();
    ret += serialization::serialized_size(host_malloc_container.num_variants);
    ret += serialization::serialized_size(host_malloc_container.chromosome);
    ret += serialization::serialized_size(host_malloc_container.chromosome_window_start);
    ret += serialization::serialized_size(host_malloc_container.reference_window_start);
    ret += serialization::serialized_size(host_malloc_container.alignment_window_len);
    ret += serialization::serialized_size(host_malloc_container.id);
    ret += serialization::serialized_size(host_malloc_container.qual);
    ret += serialization::serialized_size(host_malloc_container.n_samples);
    ret += serialization::serialized_size(host_malloc_container.n_alleles);
    // xxxnsubtil: need to fix packed vector serialization!
    ret += serialization::serialized_size(host_malloc_container.reference_sequence.m_size);
    ret += serialization::serialized_size(host_malloc_container.reference_sequence.m_storage);
    ret += serialization::serialized_size(host_malloc_container.reference_sequence_start);
    ret += serialization::serialized_size(host_malloc_container.reference_sequence_len);
    ret += serialization::serialized_size(host_malloc_container.alternate_sequence.m_size);
    ret += serialization::serialized_size(host_malloc_container.alternate_sequence.m_storage);
    ret += serialization::serialized_size(host_malloc_container.alternate_sequence_start);
    ret += serialization::serialized_size(host_malloc_container.alternate_sequence_len);

    return ret;
}

void *variant_database::serialize(void *out)
{
    out = serialization::encode(out, &data_mask);
    out = id_db.serialize(out);
    out = serialization::encode(out, &host_malloc_container.num_variants);
    out = serialization::encode(out, &host_malloc_container.chromosome);
    out = serialization::encode(out, &host_malloc_container.chromosome_window_start);
    out = serialization::encode(out, &host_malloc_container.reference_window_start);
    out = serialization::encode(out, &host_malloc_container.alignment_window_len);
    out = serialization::encode(out, &host_malloc_container.id);
    out = serialization::encode(out, &host_malloc_container.qual);
    out = serialization::encode(out, &host_malloc_container.n_samples);
    out = serialization::encode(out, &host_malloc_container.n_alleles);
    out = serialization::encode(out, &host_malloc_container.reference_sequence.m_size);
    out = serialization::encode(out, &host_malloc_container.reference_sequence.m_storage);
    out = serialization::encode(out, &host_malloc_container.reference_sequence_start);
    out = serialization::encode(out, &host_malloc_container.reference_sequence_len);
    out = serialization::encode(out, &host_malloc_container.alternate_sequence.m_size);
    out = serialization::encode(out, &host_malloc_container.alternate_sequence.m_storage);
    out = serialization::encode(out, &host_malloc_container.alternate_sequence_start);
    out = serialization::encode(out, &host_malloc_container.alternate_sequence_len);

    return out;
}

void variant_database::unserialize(shared_memory_file& shm)
{
    void *in = shm.data;

    in = serialization::decode(&data_mask, in);
    in = id_db.unserialize(in);
    in = serialization::decode(&host.num_variants, in);
    in = serialization::unwrap_vector_view(host.chromosome, in);
    in = serialization::unwrap_vector_view(host.chromosome_window_start, in);
    in = serialization::unwrap_vector_view(host.reference_window_start, in);
    in = serialization::unwrap_vector_view(host.alignment_window_len, in);
    in = serialization::unwrap_vector_view(host.id, in);
    in = serialization::unwrap_vector_view(host.qual, in);
    in = serialization::unwrap_vector_view(host.n_samples, in);
    in = serialization::unwrap_vector_view(host.n_alleles, in);
    in = serialization::unwrap_packed_vector_view(host.reference_sequence, in);
    in = serialization::unwrap_vector_view(host.reference_sequence_start, in);
    in = serialization::unwrap_vector_view(host.reference_sequence_len, in);
    in = serialization::unwrap_packed_vector_view(host.alternate_sequence, in);
    in = serialization::unwrap_vector_view(host.alternate_sequence_start, in);
    in = serialization::unwrap_vector_view(host.alternate_sequence_len, in);

    host_mmap_container = shm;
}

size_t variant_database::download(void)
{
    size_t num_bytes = 0;
#define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.f[0]) * device.f.size();
#define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * device.f.m_storage.size();

    device.num_variants = host.num_variants;

    if (data_mask & VariantDataMask::ID)
    {
        device.id.copy_from_view(host.id);
        TRACK_VECTOR_SIZE(id);
    } else {
        device.id.clear();
    }

    if (data_mask & VariantDataMask::CHROMOSOME)
    {
        device.chromosome.copy_from_view(host.chromosome);
        TRACK_VECTOR_SIZE(chromosome);
    } else {
        device.chromosome.clear();
    }

    if (data_mask & VariantDataMask::ALIGNMENT)
    {
        device.chromosome_window_start.copy_from_view(host.chromosome_window_start);
        device.reference_window_start.copy_from_view(host.reference_window_start);
        device.alignment_window_len.copy_from_view(host.alignment_window_len);

        TRACK_VECTOR_SIZE(chromosome_window_start);
        TRACK_VECTOR_SIZE(reference_window_start);
        TRACK_VECTOR_SIZE(alignment_window_len);
    } else {
        device.chromosome_window_start.clear();
        device.reference_window_start.clear();
        device.alignment_window_len.clear();
    }

    if (data_mask & VariantDataMask::QUAL)
    {
        device.qual.copy_from_view(host.qual);
        TRACK_VECTOR_SIZE(qual);
    } else {
        device.qual.clear();
    }

    if (data_mask & VariantDataMask::N_SAMPLES)
    {
        device.n_samples.copy_from_view(host.n_samples);
        TRACK_VECTOR_SIZE(n_samples);
    } else {
        device.n_samples.clear();
    }

    if (data_mask & VariantDataMask::N_ALLELES)
    {
        device.n_alleles.copy_from_view(host.n_alleles);
        TRACK_VECTOR_SIZE(n_alleles);
    } else {
        device.n_alleles.clear();
    }

    if (data_mask & VariantDataMask::REFERENCE)
    {
        device.reference_sequence.copy_from_view(host.reference_sequence);
        device.reference_sequence_start.copy_from_view(host.reference_sequence_start);
        device.reference_sequence_len.copy_from_view(host.reference_sequence_len);

        TRACK_PACKED_VECTOR_SIZE(reference_sequence);
        TRACK_VECTOR_SIZE(reference_sequence_start);
        TRACK_VECTOR_SIZE(reference_sequence_len);
    } else {
        device.reference_sequence.clear();
        device.reference_sequence_start.clear();
        device.reference_sequence_len.clear();
    }

    if (data_mask & VariantDataMask::ALTERNATE)
    {
        device.alternate_sequence.copy_from_view(host.alternate_sequence);
        device.alternate_sequence_start.copy_from_view(host.alternate_sequence_start);
        device.alternate_sequence_len.copy_from_view(host.alternate_sequence_len);

        TRACK_PACKED_VECTOR_SIZE(alternate_sequence);
        TRACK_VECTOR_SIZE(alternate_sequence_start);
        TRACK_VECTOR_SIZE(alternate_sequence_len);
    } else {
        device.alternate_sequence.clear();
        device.alternate_sequence_start.clear();
        device.alternate_sequence_len.clear();
    }

    return num_bytes;
#undef TRACK_VECTOR_SIZE
#undef TRACK_PACKED_VECTOR_SIZE
}
