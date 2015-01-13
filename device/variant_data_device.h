/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
            this->feature_start = host.feature_start;
            this->feature_stop = host.feature_stop;

            TRACK_VECTOR_SIZE(feature_start);
            TRACK_VECTOR_SIZE(feature_stop);
        } else {
            this->feature_start.clear();
            this->feature_stop.clear();
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

