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

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/io/fmi.h>

#include "util.h"
#include "reference.h"

using namespace nvbio;

reference_genome_device::reference_genome_device()
    : d_fmi(NULL)
{
}

reference_genome_device::~reference_genome_device()
{
    if (d_fmi)
    {
        delete d_fmi;
        d_fmi = NULL;
    }
}

void reference_genome_device::load(io::FMIndexData *h_fmi, const nvbio::vector<host_tag, uint64>& ref_sequence_offsets)
{
    d_fmi = new io::FMIndexDataDevice(*h_fmi, io::FMIndexData::GENOME);
    this->ref_sequence_offsets = ref_sequence_offsets;
}

reference_genome::reference_genome()
    : h_fmi(NULL)
{
}

reference_genome::~reference_genome()
{
    if (h_fmi)
    {
        delete h_fmi;
        h_fmi = NULL;
    }
}

bool reference_genome::load(const char *name)
{
    io::FMIndexDataMMAP *mmap = new io::FMIndexDataMMAP();
    if (mmap->load(name))
    {
        h_fmi = mmap;
        generate_reference_sequence_map();
        return true;
    }

    delete mmap;

    io::FMIndexDataRAM *file = new io::FMIndexDataRAM();
    if (file->load(name, io::FMIndexData::GENOME))
    {
        h_fmi = file;
        generate_reference_sequence_map();
        return true;
    }

    return false;
}

void reference_genome::download(void)
{
    device.load(h_fmi, ref_sequence_offsets);
}

void reference_genome::generate_reference_sequence_map(void)
{
    ref_sequence_offsets.resize(h_fmi->m_bnt_info.n_seqs);

    uint32 ref_seq_id = 0;
    for(unsigned int i = 0; i < h_fmi->m_bnt_info.n_seqs; i++)
    {
        io::BNTAnn *ann = &h_fmi->m_bnt_data.anns[i];
        char *name = &h_fmi->m_bnt_data.names[ann->name_offset];
        uint32 h = bqsr_string_hash(name);

        assert(ref_sequence_id_map.find(h) == ref_sequence_id_map.end()
                || !"duplicate reference sequence name!");

        ref_sequence_id_map[h] = ref_seq_id;
        ref_sequence_offsets[ref_seq_id] = ann->offset;

        ref_seq_id++;
    }
}

