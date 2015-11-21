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

#include <lift/parallel.h>

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
