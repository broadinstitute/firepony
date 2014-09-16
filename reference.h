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

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/vcf.h>

#include <map>

#include "util.h"
#include "alignment_data.h"

struct reference_genome_device
{
    nvbio::io::SequenceDataDevice *d_ref;
    D_VectorU32 sequence_offsets;

    reference_genome_device();
    ~reference_genome_device();

    void load(nvbio::io::SequenceData *h_ref, const H_VectorU32& ref_sequence_offsets);
    void transform_alignment_start_positions(alignment_batch_device *batch);

    struct const_view
    {
        const nvbio::io::SequenceDataDevice::const_plain_view_type genome_stream;
        const D_VectorU32::const_view sequence_offsets;
    };

    operator const_view() const
    {
        const_view v = {
                nvbio::plain_view(*d_ref),
                sequence_offsets,
        };

        return v;
    }
};

struct reference_genome
{
    nvbio::io::SequenceData *h_ref;
    // maps ref sequence name hash to ref sequence ID
    std::map<uint32, uint32> sequence_id_map;
    // maps ref sequence ID to ref sequence offset
    H_VectorU32 sequence_offsets;

    struct reference_genome_device device;

    reference_genome();
    ~reference_genome();

    bool load(const char *name);
    void download(void);

private:
    void generate_reference_sequence_map(void);
};
