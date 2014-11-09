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
#include "string_database.h"
#include "mmap.h"

#include <vector>

#include <gamgee/variant.h>

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

template <typename system_tag>
struct variant_database_storage
{
    template <typename T> using Vector = vector<system_tag, T>;
    template <uint32 bits> using PackedVector = packed_vector<system_tag, bits>;

    uint32 num_variants;

    Vector<uint32> chromosome;
    Vector<uint32> chromosome_window_start; // start position relative to the sequence
    Vector<uint32> reference_window_start;  // global genome start position
    Vector<uint32> alignment_window_len;    // length of the alignment window

    Vector<uint32> id;
    Vector<float> qual;
    Vector<uint32> n_samples;
    Vector<uint32> n_alleles;

    // note: we assume that each DB entry will only have one alternate_sequence
    // for VCF entries with multiple alternates, we generate one DB entry
    // for each alternate_sequence
    // also note that we don't support alternate IDs here, only sequences
    PackedVector<4> reference_sequence;
    Vector<uint32> reference_sequence_start;
    Vector<uint32> reference_sequence_len;

    PackedVector<4> alternate_sequence;
    Vector<uint32> alternate_sequence_start;
    Vector<uint32> alternate_sequence_len;

    CUDA_HOST variant_database_storage()
        : num_variants(0)
    { }

    struct const_view
    {
        uint32 num_variants;

        typename Vector<uint32>::const_view chromosome;
        typename Vector<uint32>::const_view chromosome_window_start;
        typename Vector<uint32>::const_view reference_window_start;
        typename Vector<uint32>::const_view alignment_window_len;

        typename Vector<uint32>::const_view id;
        typename Vector<float>::const_view qual;
        typename Vector<uint32>::const_view n_samples;
        typename Vector<uint32>::const_view n_alleles;

        typename PackedVector<4>::const_view reference_sequence;
        typename Vector<uint32>::const_view reference_sequence_start;
        typename Vector<uint32>::const_view reference_sequence_len;

        typename PackedVector<4>::const_view alternate_sequence;
        typename Vector<uint32>::const_view alternate_sequence_start;
        typename Vector<uint32>::const_view alternate_sequence_len;

        typename Vector<uint2>::const_view chromosome_reference_window;
    };

    CUDA_HOST operator const_view() const
    {
        struct const_view v = {
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

typedef variant_database_storage<target_system_tag> variant_database_device;
typedef variant_database_storage<host_tag> variant_database_host;

struct variant_database
{
    uint32 data_mask;

    string_database id_db;

    variant_database_host::const_view host;
    variant_database_host host_malloc_container;
    shared_memory_file host_mmap_container;

    variant_database_device device;

    size_t serialized_size(void);
    void *serialize(void *out);
    void unserialize(shared_memory_file& shm);

    size_t download(void);
};

} // namespace firepony
