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

// set to 1 to store BAQ state in the context, useful for debugging
#define PRESERVE_BAQ_STATE 0
// set to 1 to run separate kernels for the various HMM stages
#define BAQ_HMM_SPLIT_PHASE 1

template <target_system system>
struct baq_context
{
    // reference windows for HMM relative to the base alignment position
    // note: these are signed and the first coordinate is expected to be negative
    vector<system, short2> hmm_reference_windows;
    // band widths
    vector<system, uint16> bandwidth;

    // BAQ'ed qualities for each read, same size as each read
    vector<system, uint8> qualities;
#if PRESERVE_BAQ_STATE
    // BAQ state, for debugging
    vector<system, uint32> state;
#endif

    // forward and backward HMM matrices
    // each read requires read_len * 6 * (bandWidth + 1)
    vector<system, double> forward;
    vector<system, double> backward;
    // index vector for forward/backward matrices
    vector<system, uint32> matrix_index;

    // scaling factors
    vector<system, double> scaling;
    // index vector for scaling factors
    vector<system, uint32> scaling_index;

    struct view
    {
        typename vector<system, short2>::view hmm_reference_windows;
        typename vector<system, uint16>::view bandwidth;
        typename vector<system, uint8>::view qualities;
        typename vector<system, double>::view forward;
        typename vector<system, double>::view backward;
        typename vector<system, uint32>::view matrix_index;
        typename vector<system, double>::view scaling;
        typename vector<system, uint32>::view scaling_index;
    };

    operator view()
    {
        view v = {
                hmm_reference_windows,
                bandwidth,
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
