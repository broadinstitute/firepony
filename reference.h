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

#include "bam_loader.h"
#include "util.h"

using namespace nvbio;

struct reference_genome_device
{
    io::SequenceDataDevice *d_ref;
    D_VectorU32 ref_sequence_offsets;

    reference_genome_device();
    ~reference_genome_device();

    void load(io::SequenceData *h_ref, const H_VectorU32& ref_sequence_offsets);
    void transform_alignment_start_positions(BAM_alignment_batch_device *batch);

    struct const_view
    {
        const nvbio::io::SequenceDataDevice::const_plain_view_type genome_stream;
        const D_VectorU32::const_plain_view_type ref_sequence_offsets;
    };

    operator const_view() const
    {
        const_view v = {
                plain_view(*d_ref),
                plain_view(ref_sequence_offsets)
        };

        return v;
    }
};

struct reference_genome
{
    io::SequenceData *h_ref;
    // maps ref sequence name hash to ref sequence ID
    std::map<uint32, uint32> ref_sequence_id_map;
    // maps ref sequence ID to ref sequence offset
    H_VectorU32 ref_sequence_offsets;

    struct reference_genome_device device;

    reference_genome();
    ~reference_genome();

    bool load(const char *name);
    void download(void);

private:
    void generate_reference_sequence_map(void);
};
