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
#include "reference.h"

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

    static NVBIO_HOST_DEVICE char ascii(Event e)
    {
        return e == M ? 'M' :
               e == I ? 'I' :
               e == D ? 'D' :
                        'S';
    }

    static NVBIO_HOST_DEVICE char ascii(uint32 e)
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

    struct view
    {
        D_VectorU32::plain_view_type cigar_offsets;
        D_PackedVector_2b::plain_view_type cigar_events;
        D_VectorU16::plain_view_type cigar_event_read_coordinates;
        D_VectorU16::plain_view_type cigar_event_reference_coordinates;
        D_VectorU16_2::plain_view_type read_window_clipped;
        D_VectorU16_2::plain_view_type read_window_clipped_no_insertions;
        D_VectorU16_2::plain_view_type reference_window_clipped;
        D_PackedVector_1b::plain_view_type is_snp;
    };

    operator view()
    {
        view v = {
                plain_view(cigar_offsets),
                plain_view(cigar_events),
                plain_view(cigar_event_read_coordinates),
                plain_view(cigar_event_reference_coordinates),
                plain_view(read_window_clipped),
                plain_view(read_window_clipped_no_insertions),
                plain_view(reference_window_clipped),
                plain_view(is_snp),
        };

        return v;
    }
};

void expand_cigars(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_device& batch);
void debug_cigar(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_host& batch, int read_index);
