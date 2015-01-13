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

