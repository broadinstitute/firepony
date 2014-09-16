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
#include "alignment_data.h"
#include "sequence_data.h"

struct cigar_event
{
    // note that this is stored in a packed 2-bit vector
    typedef enum {
        M = 0,  // match or mismatch
        I = 1,  // insertion to the reference
        D = 2,  // deletion from the reference
        S = 3,  // soft clipping (bases exist in read but not in reference)
    } Event;

    static CUDA_HOST_DEVICE char ascii(Event e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
                        'S';
    }

    static CUDA_HOST_DEVICE char ascii(uint32 e)
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
    D_PackedVector_2b cigar_events;
    // the read coordinate for each cigar event
    D_VectorU16 cigar_event_read_coordinates;
    // the reference coordinate for each cigar event, relative to the start of the alignment window
    D_VectorU16 cigar_event_reference_coordinates;

    // alignment window in the read, not including clipped bases
    D_VectorU16_2 read_window_clipped;
    // alignment window in the read, not including clipped bases or leading/trailing insertions
    D_VectorU16_2 read_window_clipped_no_insertions;
    // alignment window in the reference, not including clipped bases (relative to base alignment position)
    D_VectorU16_2 reference_window_clipped;

    // bit vector representing SNPs, one per cigar event
    // (1 means reference mismatch, 0 means match or non-M cigar event)
    D_PackedVector_1b is_snp;

    // number of errors for each read
    D_VectorU16 num_errors;

    struct view
    {
        D_VectorU32::view cigar_offsets;
        D_PackedVector_2b::view cigar_events;
        D_VectorU16::view cigar_event_read_coordinates;
        D_VectorU16::view cigar_event_reference_coordinates;
        D_VectorU16_2::view read_window_clipped;
        D_VectorU16_2::view read_window_clipped_no_insertions;
        D_VectorU16_2::view reference_window_clipped;
        D_PackedVector_1b::view is_snp;
        D_VectorU16::view num_errors;
    };

    operator view()
    {
        view v = {
                cigar_offsets,
                cigar_events,
                cigar_event_read_coordinates,
                cigar_event_reference_coordinates,
                read_window_clipped,
                read_window_clipped_no_insertions,
                reference_window_clipped,
                is_snp,
                num_errors,
        };

        return v;
    }
};

void expand_cigars(bqsr_context *context, const alignment_batch& batch);
void debug_cigar(bqsr_context *context, const alignment_batch& batch, int read_index);
