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

namespace firepony {

#include "bit_packing.h"

// this is a little weird, since it doesn't actually change
template <target_system system, typename PreviousCovariate = covariate_null<system> >
struct covariate_EventTracker : public covariate<system, PreviousCovariate, 2>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(typename firepony_context<system>::view ctx,
                                                     const typename alignment_batch_device<system>::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        return covariate<system, PreviousCovariate, 2>::build_key(input_key,
                                                                  { cigar_event::M, cigar_event::I, cigar_event::D },
                                                                  ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

} // namespace firepony
