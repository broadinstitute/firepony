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
#include "covariates_table.h"

struct covariates_context
{
    // read window after clipping low quality ends
    D_VectorU16_2 high_quality_window;

    D_CovariateTable scratch_table_space;

    D_CovariateTable quality;
    D_CovariateTable cycle;
    D_CovariateTable context;

    struct view
    {
        D_VectorU16_2::view high_quality_window;
        D_CovariateTable::view scratch_table_space;
    };

    operator view()
    {
        view v = {
            high_quality_window,
            scratch_table_space,
        };

        return v;
    }
};

void gather_covariates(bqsr_context *context, const alignment_batch& batch);
void output_covariates(bqsr_context *context);
