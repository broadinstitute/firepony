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

#include "bqsr_types.h"
#include "util.h"

#include <vector>

#include <gamgee/variant.h>

struct variant_db_header
{
       string_database id_db;
       string_database chromosome_db;
};

namespace VariantDataMask
{
    enum
    {
        CHROMOSOME          = 0x001,
        ALIGNMENT_START     = 0x002,
        ALIGNMENT_STOP      = 0x004,
        ID                  = 0x008,
        REFERENCE           = 0x010,
        ALTERNATE           = 0x020,
        QUAL                = 0x040,
        N_SAMPLES           = 0x100,
        N_ALLELES           = 0x200
    };
};

template <typename system_tag>
struct variant_db_storage
{
    template <typename T> using Vector = bqsr::vector<system_tag, T>;
    template <uint32 bits> using PackedVector = bqsr::packed_vector<system_tag, bits>;

    uint32 num_variants;

    Vector<uint32> chromosome;
    Vector<uint32> alignment_start;
    Vector<uint32> alignment_stop;
    Vector<uint32> id;
    Vector<float> qual;
    Vector<uint32> n_samples;
    Vector<uint32> n_alleles;

    // note: we assume that each DB entry will only have one alternate
    // for VCF entries with multiple alternates, we generate one DB entry
    // for each alternate
    // also note that we don't support alternate IDs here, only sequences
    PackedVector<4> reference;
    Vector<uint32> reference_start;
    Vector<uint32> reference_len;

    PackedVector<4> alternate;
    Vector<uint32> alternate_start;
    Vector<uint32> alternate_len;

    CUDA_HOST variant_db_storage()
        : num_variants(0)
    { }
};

struct variant_db_device : public variant_db_storage<target_system_tag>
{
    struct const_view
    {
        uint32 num_variants;

        D_VectorU32::const_view chromosome;
        D_VectorU32::const_view alignment_start;
        D_VectorU32::const_view alignment_stop;
        D_VectorF32::const_view qual;
        D_VectorU32::const_view n_samples;
        D_VectorU32::const_view n_alleles;

        D_PackedVector<4>::const_view reference;
        D_VectorU32::const_view reference_start;
        D_VectorU32::const_view reference_len;

        D_PackedVector<4>::const_view alternate;
        D_VectorU32::const_view alternate_start;
        D_VectorU32::const_view alternate_len;
    };

    operator const_view() const
    {
        struct const_view v = {
                num_variants,

                chromosome,
                alignment_start,
                alignment_stop,
                qual,
                n_samples,
                n_alleles,
                reference,
                reference_start,
                reference_len,
                alternate,
                alternate_start,
                alternate_len,
        };

        return v;
    }
};

struct variant_db_host : public variant_db_storage<host_tag>
{
};

struct variant_db
{
    uint32 data_mask;

    variant_db_header header;

    variant_db_host host;
    variant_db_device device;

    void download(void)
    {
        device.num_variants = host.num_variants;

        if (data_mask & VariantDataMask::CHROMOSOME)
        {
            device.chromosome = host.chromosome;
        } else {
            device.chromosome.clear();
        }

        if (data_mask & VariantDataMask::ALIGNMENT_START)
        {
            device.alignment_start = host.alignment_start;
        } else {
            device.alignment_start.clear();
        }

        if (data_mask & VariantDataMask::ALIGNMENT_STOP)
        {
            device.alignment_stop = host.alignment_stop;
        } else {
            device.alignment_stop.clear();
        }

        if (data_mask & VariantDataMask::QUAL)
        {
            device.qual = host.qual;
        } else {
            device.qual.clear();
        }

        if (data_mask & VariantDataMask::N_SAMPLES)
        {
            device.n_samples = host.n_samples;
        } else {
            device.n_samples.clear();
        }

        if (data_mask & VariantDataMask::N_ALLELES)
        {
            device.n_alleles = host.n_alleles;
        } else {
            device.n_alleles.clear();
        }
    };
};
