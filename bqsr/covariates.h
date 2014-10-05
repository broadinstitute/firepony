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
    D_CovariatePool mempool;

    D_CovariateTable recal_table_1;
    D_CovariateTable cycle;
    D_CovariateTable context;

    struct view
    {
        D_CovariatePool::view mempool;
//        D_CovariateTable::view recal_table_1;
    };

    operator view()
    {
        view v = {
            mempool,
//            recal_table_1,
        };

        return v;
    }
};

void gather_covariates(bqsr_context *context, const alignment_batch& batch);
void output_covariates(bqsr_context *context);
