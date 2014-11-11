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

#include "device_types.h"
#include "alignment_data_device.h"

namespace firepony {

#define NO_BAQ_UNCERTAINTY 64

template <target_system system>
struct baq_context
{
    // reference windows for HMM
    d_vector_u32_2<system> reference_windows;

    // BAQ'ed qualities for each read, same size as each read
    d_vector_u8<system> qualities;

    // forward and backward HMM matrices
    // each read requires read_len * 6 * (bandWidth + 1)
    d_vector<system, double> forward;
    d_vector<system, double> backward;
    // index vector for forward/backward matrices
    d_vector<system, uint32> matrix_index;

    // scaling factors
    d_vector<system, double> scaling;
    // index vector for scaling factors
    d_vector<system, uint32> scaling_index;

    struct view
    {
        typename d_vector_u32_2<system>::view reference_windows;
        typename d_vector_u8<system>::view qualities;
        typename d_vector<system, double>::view forward;
        typename d_vector<system, double>::view backward;
        typename d_vector<system, uint32>::view matrix_index;
        typename d_vector<system, double>::view scaling;
        typename d_vector<system, uint32>::view scaling_index;
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

template <target_system system> void baq_reads(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void debug_baq(firepony_context<system>& context, const alignment_batch<system>& batch, int read_index);

} // namespace firepony
