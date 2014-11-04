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

namespace firepony {

struct fractional_error_context
{
    D_VectorF64 snp_errors;
    D_VectorF64 insertion_errors;
    D_VectorF64 deletion_errors;

    struct view
    {
        D_VectorF64::view snp_errors;
        D_VectorF64::view insertion_errors;
        D_VectorF64::view deletion_errors;
    };

    operator view()
    {
        view v = {
            snp_errors,
            insertion_errors,
            deletion_errors,
        };

        return v;
    }
};

void build_fractional_error_arrays(bqsr_context *ctx, const alignment_batch& batch);
void debug_fractional_error_arrays(bqsr_context *context, const alignment_batch& batch, int read_index);

} // namespace firepony
