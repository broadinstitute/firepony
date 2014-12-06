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

template <target_system system>
struct variant_database_device : public variant_database_storage<system>
{
    size_t download(const variant_database_host& host)
    {
        size_t num_bytes = 0;
    #define TRACK_VECTOR_SIZE(f) num_bytes += sizeof(host.f[0]) * this->f.size();
    #define TRACK_PACKED_VECTOR_SIZE(f) num_bytes += sizeof(uint32) * this->f.m_storage.size();

        this->data_mask = host.data_mask;
        this->num_variants = host.num_variants;

        if (this->data_mask & VariantDataMask::ID)
        {
            this->id = host.id;
            TRACK_VECTOR_SIZE(id);
        } else {
            this->id.clear();
        }

        if (this->data_mask & VariantDataMask::CHROMOSOME)
        {
            this->chromosome = host.chromosome;
            TRACK_VECTOR_SIZE(chromosome);
        } else {
            this->chromosome.clear();
        }

        if (this->data_mask & VariantDataMask::ALIGNMENT)
        {
            this->chromosome_window_start = host.chromosome_window_start;
            this->reference_window_start = host.reference_window_start;
            this->alignment_window_len = host.alignment_window_len;

            TRACK_VECTOR_SIZE(chromosome_window_start);
            TRACK_VECTOR_SIZE(reference_window_start);
            TRACK_VECTOR_SIZE(alignment_window_len);
        } else {
            this->chromosome_window_start.clear();
            this->reference_window_start.clear();
            this->alignment_window_len.clear();
        }

        if (this->data_mask & VariantDataMask::QUAL)
        {
            this->qual = host.qual;
            TRACK_VECTOR_SIZE(qual);
        } else {
            this->qual.clear();
        }

        if (this->data_mask & VariantDataMask::N_SAMPLES)
        {
            this->n_samples = host.n_samples;
            TRACK_VECTOR_SIZE(n_samples);
        } else {
            this->n_samples.clear();
        }

        if (this->data_mask & VariantDataMask::N_ALLELES)
        {
            this->n_alleles = host.n_alleles;
            TRACK_VECTOR_SIZE(n_alleles);
        } else {
            this->n_alleles.clear();
        }

        if (this->data_mask & VariantDataMask::REFERENCE)
        {
            this->reference_sequence = host.reference_sequence;
            this->reference_sequence_start = host.reference_sequence_start;
            this->reference_sequence_len = host.reference_sequence_len;

            TRACK_PACKED_VECTOR_SIZE(reference_sequence);
            TRACK_VECTOR_SIZE(reference_sequence_start);
            TRACK_VECTOR_SIZE(reference_sequence_len);
        } else {
            this->reference_sequence.clear();
            this->reference_sequence_start.clear();
            this->reference_sequence_len.clear();
        }

        if (this->data_mask & VariantDataMask::ALTERNATE)
        {
            this->alternate_sequence = host.alternate_sequence;
            this->alternate_sequence_start = host.alternate_sequence_start;
            this->alternate_sequence_len = host.alternate_sequence_len;

            TRACK_PACKED_VECTOR_SIZE(alternate_sequence);
            TRACK_VECTOR_SIZE(alternate_sequence_start);
            TRACK_VECTOR_SIZE(alternate_sequence_len);
        } else {
            this->alternate_sequence.clear();
            this->alternate_sequence_start.clear();
            this->alternate_sequence_len.clear();
        }

        return num_bytes;
    #undef TRACK_VECTOR_SIZE
    #undef TRACK_PACKED_VECTOR_SIZE
    }
};

template <target_system system>
struct variant_database
{
    const variant_database_host& host;
    variant_database_device<system> device;

    variant_database(const variant_database_host& host)
        : host(host)
    { }

    size_t download(void)
    {
        return device.download(host);
    }
};

} // namespace firepony

