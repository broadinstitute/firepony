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

#include <vector>

namespace firepony {

namespace VariantDataMask
{
    enum
    {
        CHROMOSOME          = 0x001,
        ALIGNMENT           = 0x002,
        ID                  = 0x008,
        REFERENCE           = 0x010,
        ALTERNATE           = 0x020,
        QUAL                = 0x040,
        N_SAMPLES           = 0x100,
        N_ALLELES           = 0x200
    };
};

template <target_system system>
struct variant_database_storage
{
    uint32 data_mask;
    uint32 num_variants;

    vector<system, uint32> chromosome;
    vector<system, uint32> chromosome_window_start; // start position relative to the sequence
    vector<system, uint32> reference_window_start;  // global genome start position
    vector<system, uint32> alignment_window_len;    // length of the alignment window

    vector<system, uint32> id;
    vector<system, float> qual;
    vector<system, uint32> n_samples;
    vector<system, uint32> n_alleles;

    // note: we assume that each DB entry will only have one alternate_sequence
    // for VCF entries with multiple alternates, we generate one DB entry
    // for each alternate_sequence
    // also note that we don't support alternate IDs here, only sequences
    packed_vector<system, 4> reference_sequence;
    vector<system, uint32> reference_sequence_start;
    vector<system, uint32> reference_sequence_len;

    packed_vector<system, 4> alternate_sequence;
    vector<system, uint32> alternate_sequence_start;
    vector<system, uint32> alternate_sequence_len;

    CUDA_HOST variant_database_storage()
        : num_variants(0)
    { }

    struct const_view
    {
        uint32 data_mask;
        uint32 num_variants;

        typename vector<system, uint32>::const_view chromosome;
        typename vector<system, uint32>::const_view chromosome_window_start;
        typename vector<system, uint32>::const_view reference_window_start;
        typename vector<system, uint32>::const_view alignment_window_len;

        typename vector<system, uint32>::const_view id;
        typename vector<system, float>::const_view qual;
        typename vector<system, uint32>::const_view n_samples;
        typename vector<system, uint32>::const_view n_alleles;

        typename packed_vector<system, 4>::const_view reference_sequence;
        typename vector<system, uint32>::const_view reference_sequence_start;
        typename vector<system, uint32>::const_view reference_sequence_len;

        typename packed_vector<system, 4>::const_view alternate_sequence;
        typename vector<system, uint32>::const_view alternate_sequence_start;
        typename vector<system, uint32>::const_view alternate_sequence_len;
    };

    CUDA_HOST operator const_view() const
    {
        struct const_view v = {
                data_mask,
                num_variants,

                chromosome,
                chromosome_window_start,
                reference_window_start,
                alignment_window_len,

                id,
                qual,
                n_samples,
                n_alleles,
                reference_sequence,
                reference_sequence_start,
                reference_sequence_len,
                alternate_sequence,
                alternate_sequence_start,
                alternate_sequence_len,
        };

        return v;
    }
};

struct variant_database_host : public variant_database_storage<host>
{
    string_database id_db;
};

} // namespace firepony

