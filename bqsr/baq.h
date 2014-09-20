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

#define NO_BAQ_UNCERTAINTY 64

struct baq_context
{
    // reference windows for HMM
    D_VectorU32_2 reference_windows;

    // BAQ'ed qualities for each read, same size as each read
    D_VectorU8 qualities;

    // forward and backward HMM matrices
    // each read requires read_len * 6 * (bandWidth + 1)
    D_VectorF64 forward;
    D_VectorF64 backward;
    // index vector for forward/backward matrices
    D_VectorU32 matrix_index;

    // scaling factors
    D_VectorF64 scaling;
    // index vector for scaling factors
    D_VectorU32 scaling_index;

    struct view
    {
        D_VectorU32_2::view reference_windows;
        D_VectorU8::view qualities;
        D_VectorF64::view forward;
        D_VectorF64::view backward;
        D_VectorU32::view matrix_index;
        D_VectorF64::view scaling;
        D_VectorU32::view scaling_index;
    };

    operator view()
    {
        view v = {
                reference_windows,
                qualities,
                forward,
                backward,
                matrix_index,
                scaling,
                scaling_index,
        };

        return v;
    }
};

void baq_reads(bqsr_context *context, const alignment_batch& batch);
void debug_baq(bqsr_context *context, const alignment_batch& batch, int read_index);
