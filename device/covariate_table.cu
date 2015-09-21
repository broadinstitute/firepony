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

#include "../types.h"
#include "firepony_context.h"
#include "covariates.h"
#include "covariate_table.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>

#include <lift/parallel.h>

namespace firepony {

template <target_system system, typename covariate_value>
void covariate_table<system, covariate_value>::sort(vector<system, covariate_key>& temp_keys,
                                                    vector<system, covariate_value>& temp_values,
                                                    vector<system, uint8>& temp_storage,
                                                    uint32 num_key_bits)
{
    if (this->keys.size())
    {
        parallel<system>::sort_by_key(this->keys,
                                      this->values,
                                      temp_keys,
                                      temp_values,
                                      temp_storage,
                                      num_key_bits);
    }
}

METHOD_INSTANTIATE(covariate_observation_table, sort);
METHOD_INSTANTIATE(covariate_empirical_table, sort);

template <typename covariate_value>
struct covariate_value_sum
{ };

template <>
struct covariate_value_sum<covariate_observation_value>
{
    CUDA_HOST_DEVICE covariate_observation_value operator() (const covariate_observation_value& a, const covariate_observation_value& b)
    {
        return covariate_observation_value { a.observations + b.observations,
                                             a.mismatches + b.mismatches };
    }
};

template <>
struct covariate_value_sum<covariate_empirical_value>
{
    CUDA_HOST_DEVICE covariate_empirical_value operator() (const covariate_empirical_value& a, const covariate_empirical_value& b)
    {
        return { a.observations + b.observations,
                 a.mismatches + b.mismatches,
                 a.expected_errors + b.expected_errors,
                 0.0f,
                 0.0f };
    }
};

template <target_system system, typename covariate_value>
void covariate_table<system, covariate_value>::pack(vector<system, covariate_key>& temp_keys,
                                                    vector<system, covariate_value>& temp_values,
                                                    vector<system,uint8>& temp_storage)
{
    temp_keys.resize(this->size());
    temp_values.resize(this->size());

    uint32 new_size = parallel<system>::reduce_by_key(keys, values, temp_keys, temp_values, temp_storage, covariate_value_sum<covariate_value>());

    this->keys = temp_keys;
    this->values = temp_values;

    this->resize(new_size);
}
METHOD_INSTANTIATE(covariate_observation_table, pack);
METHOD_INSTANTIATE(covariate_empirical_table, pack);

struct convert_observation_to_empirical
{
    CUDA_HOST_DEVICE covariate_empirical_value operator() (const covariate_observation_value& in)
    {
        return { in.observations,
                 in.mismatches,
                 0.0,
                 0.0,
                 0.0 };
   }
};

template <target_system system>
void covariate_observation_to_empirical_table(firepony_context<system>& context,
                                              const covariate_observation_table<system>& observation_table,
                                              covariate_empirical_table<system>& empirical_table)
{
    // resize the output table
    empirical_table.resize(observation_table.size());
    // copy the keys
    empirical_table.keys = observation_table.keys;
    // transform the values
    thrust::transform(observation_table.values.begin(),
                      observation_table.values.end(),
                      empirical_table.values.begin(),
                      convert_observation_to_empirical());
}
INSTANTIATE(covariate_observation_to_empirical_table);

} // namespace firepony
