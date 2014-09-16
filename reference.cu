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
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_access.h>
#include <nvbio/io/sequence/sequence_mmap.h>
#include <nvbio/strings/alphabet.h>

#include "util.h"
#include "reference.h"

using namespace nvbio;

reference_genome_device::reference_genome_device()
    : d_ref(NULL)
{
}

reference_genome_device::~reference_genome_device()
{
    if (d_ref)
    {
        delete d_ref;
        d_ref = NULL;
    }
}

void reference_genome_device::load(io::SequenceData *h_ref, const H_VectorU32& ref_sequence_offsets)
{
    d_ref = new io::SequenceDataDevice(*h_ref);
    this->sequence_offsets = ref_sequence_offsets;
}

reference_genome::reference_genome()
    : h_ref(NULL)
{
}

reference_genome::~reference_genome()
{
    if (h_ref)
    {
        delete h_ref;
        h_ref = NULL;
    }
}

bool reference_genome::load(const char *name)
{
    h_ref = io::map_sequence_file(name);
    if (h_ref == NULL)
        h_ref = io::load_sequence_file(DNA, name);

    if (!h_ref)
        return false;

    generate_reference_sequence_map();

    return true;
}

void reference_genome::download(void)
{
    device.load(h_ref, sequence_offsets);
}

void reference_genome::generate_reference_sequence_map(void)
{
    io::SequenceDataView view = plain_view(*h_ref);
    sequence_offsets.resize(view.m_n_seqs + 1);

    uint32 ref_seq_id = 0;
    for(unsigned int i = 0; i < view.m_n_seqs; i++)
    {
        char *name = &view.m_name_stream[view.m_name_index[i]];
        uint32 h = bqsr_string_hash(name);

        NVBIO_CUDA_ASSERT(sequence_id_map.find(h) == sequence_id_map.end() ||
                          !"duplicate reference sequence name!");

        sequence_id_map[h] = ref_seq_id;
        sequence_offsets[ref_seq_id] = view.m_sequence_index[i];

        ref_seq_id++;
    }

    sequence_offsets[view.m_n_seqs] = h_ref->m_sequence_stream_len;
}
