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
#include "covariate_table.h"

namespace firepony {

template <target_system system>
struct covariates_context
{
    // read window after clipping low quality ends
    d_vector_u16_2<system> high_quality_window;

    covariate_observation_table<system> scratch_table_space;

    covariate_observation_table<system> quality;
    covariate_observation_table<system> cycle;
    covariate_observation_table<system> context;

    covariate_empirical_table<system> read_group;

    struct view
    {
        typename d_vector_u16_2<system>::view high_quality_window;
        typename covariate_observation_table<system>::view scratch_table_space;
        typename covariate_observation_table<system>::view quality;
        typename covariate_empirical_table<system>::view read_group;
    };

    operator view()
    {
        view v = {
            high_quality_window,
            scratch_table_space,
            quality,
            read_group,
        };

        return v;
    }
};

template <target_system system> void gather_covariates(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void output_covariates(firepony_context<system>& context);

} // namespace firepony

