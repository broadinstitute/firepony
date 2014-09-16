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

#include "variant_data.h"

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
