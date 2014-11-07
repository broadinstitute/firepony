/* Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
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

#include "bit_packing.h"

namespace firepony {

template <typename PreviousCovariate = covariate_null>
struct covariate_Cycle_Illumina : public covariate<PreviousCovariate, 10, true>
{
    typedef covariate<PreviousCovariate, 10, true> base;

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

    static CUDA_HOST_DEVICE covariate_key_set encode(context::view ctx,
                                                     const alignment_batch_device::const_view batch,
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
