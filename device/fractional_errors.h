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

namespace firepony {

template <target_system system>
struct fractional_error_context
{
    vector<system, double> snp_errors;
    vector<system, double> insertion_errors;
    vector<system, double> deletion_errors;

    struct view
    {
        typename vector<system, double>::view snp_errors;
        typename vector<system, double>::view insertion_errors;
        typename vector<system, double>::view deletion_errors;
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

template <target_system system> void build_fractional_error_arrays(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void debug_fractional_error_arrays(firepony_context<system>& context, const alignment_batch<system>& batch, int read_index);

} // namespace firepony

