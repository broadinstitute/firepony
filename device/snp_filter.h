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
#include "alignment_data_device.h"
#include "sequence_data_device.h"
#include "variant_data_device.h"
#include "util.h"

namespace firepony {

template <target_system system>
struct snp_filter_context
{
    vector<system, uint32> feature_stop_sort_order; // sorted order for feature_stop

    // active reads for the VCF search
    d_vector_u32<system> active_read_ids;
    // active VCF range for each read
    d_vector_u32_2<system> active_vcf_ranges;

    struct view
    {
        typename vector<system, uint32>::view feature_stop_sort_order;

        typename d_vector_u32<system>::view active_read_ids;
        typename d_vector_u32_2<system>::view active_vcf_ranges;
    };

    operator view()
    {
        view v = {
            feature_stop_sort_order,

            active_read_ids,
            active_vcf_ranges
        };
        return v;
    }
};

template <target_system system> void build_read_offset_list(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void build_alignment_windows(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void filter_known_snps(struct firepony_context<system>& context, const alignment_batch<system>& batch);

} // namespace firepony

