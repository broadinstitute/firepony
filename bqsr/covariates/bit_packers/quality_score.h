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

template <typename PreviousCovariate = covariate_null>
struct covariate_QualityScore : public covariate<PreviousCovariate, 8>
{
    static CUDA_HOST_DEVICE covariate_key_set encode(bqsr_context::view ctx,
                                                     const alignment_batch_device::const_view batch,
                                                     uint32 read_index, uint16 bp_offset, uint32 cigar_event_index,
                                                     covariate_key_set input_key = {0, 0, 0})
    {
        const CRQ_index idx = batch.crq_index(read_index);
        // xxxnsubtil: 45 is the "default" base quality for when insertion/deletion qualities are not available in the read
        // we should eventually grab these from the alignment data itself if they're present
        covariate_key_set key = { batch.qualities[idx.qual_start + bp_offset],
                                  45,
                                  45 };

        return covariate<PreviousCovariate, 8>::build_key(input_key, key,
                                                          ctx, batch, read_index, bp_offset, cigar_event_index);
    }
};
