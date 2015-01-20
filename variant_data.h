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

    // variant data is sorted by the start position of the feature in the reference
    // (this is implicit due to the fact that VCF files are sorted by chromosome,
    // then by sequence position --- we assume that the reference chromosomes are sorted
    // in the same order)

    vector<system, uint32> chromosome;              // chromosome identifier
    vector<system, uint32> feature_start;           // feature start position in the reference
    vector<system, uint32> feature_stop;            // feature stop position in the reference

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

    // contains a prefix scan of the end points using max as the operator
    // each element encodes the maximum end point of any feature to the left of it
    vector<system, uint32> max_end_point_left;

    CUDA_HOST variant_database_storage()
        : num_variants(0)
    { }

    struct const_view
    {
        uint32 data_mask;
        uint32 num_variants;

        typename vector<system, uint32>::const_view chromosome;
        typename vector<system, uint32>::const_view feature_start;
        typename vector<system, uint32>::const_view feature_stop;

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

        typename vector<system, uint32>::const_view max_end_point_left;
    };

    CUDA_HOST operator const_view() const
    {
        struct const_view v = {
                data_mask,
                num_variants,

                chromosome,
                feature_start,
                feature_stop,

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

                max_end_point_left,
        };

        return v;
    }
};

struct variant_database_host : public variant_database_storage<host>
{
    string_database id_db;
};

} // namespace firepony

