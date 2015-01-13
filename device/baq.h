/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
