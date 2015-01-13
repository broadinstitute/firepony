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

#include "bit_packing.h"

namespace firepony {

template <target_system system, typename PreviousCovariate = covariate_null<system> >
struct covariate_Cycle_Illumina : public covariate<system, PreviousCovariate, 10, true>
{
    typedef covariate<system, PreviousCovariate, 10, true> base;

    enum {
        CUSHION_FOR_INDELS = 4
    };

    static CUDA_HOST_DEVICE covariate_key keyFromCycle(const int cycle)
    {
        // no negative values because values must fit into the first few bits of the long
        covariate_key result = abs(cycle);
//        assert(result <= (base::max_value >> 1));

        // xxxnsubtil: investigate if we can do sign propagation here instead
        result = result << 1; // shift so we can add the "sign" bit
        if (cycle < 0)
            result++;

        return result;
    }

    static CUDA_HOST_DEVICE covariate_key_set encode(typename firepony_context<system>::view ctx,
                                                     const typename alignment_batch_device<system>::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const bool paired = batch.flags[read_index] & AlignmentFlags::PAIRED;
        const bool second_of_pair = batch.flags[read_index] & AlignmentFlags::READ2;
        const bool negative_strand = batch.flags[read_index] & AlignmentFlags::REVERSE;

        const auto& window = ctx.cigar.read_window_clipped[read_index];
        const int readLength = window.y - window.x + 1;
        const int readOrderFactor = (paired && second_of_pair) ? -1 : 1;
        const int i = bp_offset - window.x;

        int cycle;
        int increment;

        if (negative_strand)
        {
            cycle = readLength * readOrderFactor;
            increment = -1 * readOrderFactor;
        } else {
            cycle = readOrderFactor;
            increment = readOrderFactor;
        }

        cycle += i * increment;

        const int MAX_CYCLE_FOR_INDELS = readLength - CUSHION_FOR_INDELS - 1;

        const covariate_key substitutionKey = keyFromCycle(cycle);
        const covariate_key indelKey = (i < CUSHION_FOR_INDELS || i > MAX_CYCLE_FOR_INDELS) ? base::invalid_key_pattern : substitutionKey;

        return base::build_key(input_key, { substitutionKey, indelKey, indelKey },
                               ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

} // namespace firepony
