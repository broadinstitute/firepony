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

#include "../types.h"
#include "covariates_table.h"

namespace firepony {

template <target_system system>
struct covariates_context
{
    // read window after clipping low quality ends
    d_vector_u16_2<system> high_quality_window;

    d_covariate_table<system> scratch_table_space;

    d_covariate_table<system> quality;
    d_covariate_table<system> cycle;
    d_covariate_table<system> context;

    struct view
    {
        typename d_vector_u16_2<system>::view high_quality_window;
        typename d_covariate_table<system>::view scratch_table_space;
        typename d_covariate_table<system>::view quality;
    };

    operator view()
    {
        view v = {
            high_quality_window,
            scratch_table_space,
            quality,
        };

        return v;
    }
};

template <target_system system> void gather_covariates(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void output_covariates(firepony_context<system>& context);

} // namespace firepony

