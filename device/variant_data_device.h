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

#include "../variant_data.h"
#include "device_types.h"

namespace firepony {

struct variant_database_device : public variant_database_storage<target_system_tag>
{
    size_t download(const variant_database_host& host)
    {
        size_t num_bytes = 0;
    #define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.view.f[0]) * f.size();
    #define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * f.m_storage.size();

        data_mask = host.view.data_mask;
        num_variants = host.view.num_variants;

        if (data_mask & VariantDataMask::ID)
        {
            id.copy_from_view(host.view.id);
            TRACK_VECTOR_SIZE(id);
        } else {
            id.clear();
        }

        if (data_mask & VariantDataMask::CHROMOSOME)
        {
            chromosome.copy_from_view(host.view.chromosome);
            TRACK_VECTOR_SIZE(chromosome);
        } else {
            chromosome.clear();
        }

        if (data_mask & VariantDataMask::ALIGNMENT)
        {
            chromosome_window_start.copy_from_view(host.view.chromosome_window_start);
            reference_window_start.copy_from_view(host.view.reference_window_start);
            alignment_window_len.copy_from_view(host.view.alignment_window_len);

            TRACK_VECTOR_SIZE(chromosome_window_start);
            TRACK_VECTOR_SIZE(reference_window_start);
            TRACK_VECTOR_SIZE(alignment_window_len);
        } else {
            chromosome_window_start.clear();
            reference_window_start.clear();
            alignment_window_len.clear();
        }

        if (data_mask & VariantDataMask::QUAL)
        {
            qual.copy_from_view(host.view.qual);
            TRACK_VECTOR_SIZE(qual);
        } else {
            qual.clear();
        }

        if (data_mask & VariantDataMask::N_SAMPLES)
        {
            n_samples.copy_from_view(host.view.n_samples);
            TRACK_VECTOR_SIZE(n_samples);
        } else {
            n_samples.clear();
        }

        if (data_mask & VariantDataMask::N_ALLELES)
        {
            n_alleles.copy_from_view(host.view.n_alleles);
            TRACK_VECTOR_SIZE(n_alleles);
        } else {
            n_alleles.clear();
        }

        if (data_mask & VariantDataMask::REFERENCE)
        {
            reference_sequence.copy_from_view(host.view.reference_sequence);
            reference_sequence_start.copy_from_view(host.view.reference_sequence_start);
            reference_sequence_len.copy_from_view(host.view.reference_sequence_len);

            TRACK_PACKED_VECTOR_SIZE(reference_sequence);
            TRACK_VECTOR_SIZE(reference_sequence_start);
            TRACK_VECTOR_SIZE(reference_sequence_len);
        } else {
            reference_sequence.clear();
            reference_sequence_start.clear();
            reference_sequence_len.clear();
        }

        if (data_mask & VariantDataMask::ALTERNATE)
        {
            alternate_sequence.copy_from_view(host.view.alternate_sequence);
            alternate_sequence_start.copy_from_view(host.view.alternate_sequence_start);
            alternate_sequence_len.copy_from_view(host.view.alternate_sequence_len);

            TRACK_PACKED_VECTOR_SIZE(alternate_sequence);
            TRACK_VECTOR_SIZE(alternate_sequence_start);
            TRACK_VECTOR_SIZE(alternate_sequence_len);
        } else {
            alternate_sequence.clear();
            alternate_sequence_start.clear();
            alternate_sequence_len.clear();
        }

        return num_bytes;
    #undef TRACK_VECTOR_SIZE
    #undef TRACK_PACKED_VECTOR_SIZE
    }
};

struct variant_database
{
    const variant_database_host& host;
    variant_database_device device;

    variant_database(const variant_database_host& host)
        : host(host)
    { }

    size_t download(void)
    {
        return device.download(host);
    }
};

} // namespace firepony

