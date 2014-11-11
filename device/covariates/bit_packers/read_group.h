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

template<target_system system, typename PreviousCovariate = covariate_null<system> >
struct covariate_ReadGroup : public covariate<system, PreviousCovariate, 8>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(typename firepony_context<system>::view ctx,
                                                     const typename alignment_batch_device<system>::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        uint32 read_group = batch.read_group[read_index];
        return covariate<system, PreviousCovariate, 8>::build_key(input_key,
                                                                  { read_group, read_group, read_group },
                                                                  ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};

} // namespace firepony
