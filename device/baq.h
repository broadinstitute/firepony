/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    persistent_allocation<system, short2> hmm_reference_windows;
    // band widths
    persistent_allocation<system, uint16> bandwidth;

    // BAQ'ed qualities for each read, same size as each read
    persistent_allocation<system, uint8> qualities;
#if PRESERVE_BAQ_STATE
    // BAQ state, for debugging
    persistent_allocation<system, uint32> state;
#endif

    // forward and backward HMM matrices
    // each read requires read_len * 6 * (bandWidth + 1)
    persistent_allocation<system, double> forward;
    persistent_allocation<system, double> backward;
    // index vector for forward/backward matrices
    persistent_allocation<system, uint32> matrix_index;

    // scaling factors
    persistent_allocation<system, double> scaling;
    // index vector for scaling factors
    persistent_allocation<system, uint32> scaling_index;

    struct view
    {
        persistent_allocation<system, short2> hmm_reference_windows;
        persistent_allocation<system, uint16> bandwidth;
        persistent_allocation<system, uint8> qualities;
        persistent_allocation<system, double> forward;
        persistent_allocation<system, double> backward;
        persistent_allocation<system, uint32> matrix_index;
        persistent_allocation<system, double> scaling;
        persistent_allocation<system, uint32> scaling_index;
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
