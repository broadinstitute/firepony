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
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/vcf.h>
#include <nvbio/io/fmi.h>

#include <map>

#include "util.h"

using namespace nvbio;

struct reference_genome_device
{
    io::FMIndexDataDevice *d_fmi;
    nvbio::vector<device_tag, uint64> ref_sequence_offsets;

    reference_genome_device();
    ~reference_genome_device();

    void load(io::FMIndexData *h_fmi, const nvbio::vector<host_tag, uint64>& ref_sequence_offsets);
};

struct reference_genome
{
    io::FMIndexData *h_fmi;
    // maps ref sequence name hash to ref sequence ID
    std::map<uint32, uint32> ref_sequence_id_map;
    // maps ref sequence ID to ref sequence offset
    nvbio::vector<host_tag, uint64> ref_sequence_offsets;

    struct reference_genome_device device;

    reference_genome();
    ~reference_genome();

    bool load(const char *name);
    void download(void);

private:
    void generate_reference_sequence_map(void);
};
