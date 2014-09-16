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

#include "../bqsr_types.h"

#include <vector>
#include <string>

#pragma once

namespace from_nvbio {

struct SNP_sequence_index
{
    // these indices are stored in base-pairs since variants are extremely short
    uint32 reference_start;
    uint32 reference_len;
    uint32 variant_start;
    uint32 variant_len;

    SNP_sequence_index()
        : reference_start(0), reference_len(0),
          variant_start(0), variant_len(0)
    { }

    SNP_sequence_index(uint32 reference_start, uint32 reference_len,
                       uint32 variant_start, uint32 variant_len)
        : reference_start(reference_start), reference_len(reference_len),
          variant_start(variant_start), variant_len(variant_len)
    { }
};

struct SNPDatabase
{
    // the name of the reference sequence
    // note: VCF allows this to be an integer ID encoded in a string that references
    // a contig from an assembly referenced in the header; this is not supported yet
    H_Vector<std::string> reference_sequence_names;

    // start (x) and stop (y) positions of the variant in the reference sequence (first base in the sequence is position 1)
    // the "stop" position is either start + len or the contents of the END= info tag
    H_VectorU32_2 sequence_positions;

    // packed reference sequences
    H_PackedVector<4> reference_sequences;
    // packed variant sequences
    H_PackedVector<4> variants;
    // an index for both references and variants
    H_Vector<SNP_sequence_index> ref_variant_index;

    // quality value assigned to each variant
    H_VectorU8 variant_qualities;

    SNPDatabase();
};

// loads variant data from file_name and appends to output
bool loadVCF(SNPDatabase& output, const char *file_name);

} // namespace from_nvbio
