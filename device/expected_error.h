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

#include "primitives/parallel.h"
#include "firepony_context.h"
#include "covariate_table.h"

namespace firepony {

// packer is one of the packer_* types that was used to create the keys in the table
template <target_system system, typename packer>
struct calc_expected_error
{
    typename covariate_empirical_table<system>::view table;

    calc_expected_error(typename covariate_empirical_table<system>::view table)
        : table(table)
    { }

    CUDA_HOST_DEVICE double qualToErrorProb(uint8 qual)
    {
        return pow(10.0, qual / -10.0);
    }

    CUDA_HOST_DEVICE double calcExpectedErrors(const covariate_empirical_value& val, uint8 qual)
    {
        return val.observations * qualToErrorProb(qual);
    }

    CUDA_HOST_DEVICE void operator() (const uint32 index)
    {
        auto& key = table.keys[index];
        auto& value = table.values[index];

        // decode the quality
        const auto qual = packer::decode(key, packer::QualityScore);

        // compute the expected error rate
        value.expected_errors = calcExpectedErrors(value, qual);
    }
};

template <target_system system, typename packer>
inline void compute_expected_error(firepony_context<system>& context, covariate_empirical_table<system>& table)
{
    parallel<system>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + table.size(),
                               calc_expected_error<system, packer>(table));
}

} // namespace firepony
