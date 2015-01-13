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

    covariate_empirical_table<system> empirical_quality;
    covariate_empirical_table<system> read_group;

    struct view
    {
        typename d_vector_u16_2<system>::view high_quality_window;
        typename covariate_observation_table<system>::view scratch_table_space;
        typename covariate_observation_table<system>::view quality;
        typename covariate_empirical_table<system>::view empirical_quality;
        typename covariate_empirical_table<system>::view read_group;
    };

    operator view()
    {
        view v = {
            high_quality_window,
            scratch_table_space,
            quality,
            empirical_quality,
            read_group,
        };

        return v;
    }
};

template <target_system system> void gather_covariates(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void postprocess_covariates(firepony_context<system>& context);
template <target_system system> void output_covariates(firepony_context<system>& context);
template <target_system system> void build_empirical_quality_score_table(firepony_context<system>& context);

} // namespace firepony

