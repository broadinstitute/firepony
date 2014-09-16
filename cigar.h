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
#include "bam_loader.h"

using namespace nvbio;

struct cigar_event
{
    // note that this is stored in a packed 2-bit vector
    typedef enum {
        M = 0,  // match or mismatch
        I = 1,  // insertion to the reference
        D = 2,  // deletion from the reference
        S = 3,  // soft clipping (bases exist in read but not in reference)
    } Event;

    static char ascii(Event e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
                        'S';
    }

    static char ascii(uint32 e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
               e == S ? 'S' :
                        '?';
    }
};

struct cigar_context
{
    // prefix sum of cigar offsets
    D_VectorU32 cigar_offsets;

    // a vector of cigar "events"
    D_PackedVector_2b cigar_ops;
    // the read coordinate for each cigar event
    D_VectorU16 cigar_op_read_coordinates;
    // the reference coordinate for each cigar event, relative to the start of the alignment window
    D_VectorU16 cigar_op_reference_coordinates;

    D_PackedVector_1b snp;
    D_PackedVector_1b indel;

    struct view
    {
        D_VectorU32::plain_view_type cigar_offsets;
        D_PackedVector_2b::plain_view_type cigar_ops;
        D_VectorU16::plain_view_type cigar_op_read_coordinates;
        D_VectorU16::plain_view_type cigar_op_reference_coordinates;
    };

    operator view()
    {
        view v = {
                plain_view(cigar_offsets),
                plain_view(cigar_ops),
                plain_view(cigar_op_read_coordinates),
                plain_view(cigar_op_reference_coordinates),
        };

        return v;
    }
};

void expand_cigars(bqsr_context *context, const BAM_alignment_batch_device& batch);
