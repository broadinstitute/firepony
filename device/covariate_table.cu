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


#include "../types.h"
#include "firepony_context.h"
#include "covariates.h"
#include "covariate_table.h"

#include "primitives/parallel.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>

namespace firepony {

template <target_system system, typename covariate_value>
void covariate_table<system, covariate_value>::sort(vector<system, covariate_key>& temp_keys,
                                                    vector<system, covariate_value>& temp_values,
                                                    vector<system, uint8>& temp_storage,
                                                    uint32 num_key_bits)
{
    parallel<system>::sort_by_key(this->keys,
                                  this->values,
                                  temp_keys,
                                  temp_values,
                                  temp_storage,
                                  num_key_bits);
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
                                                    vector<system, covariate_value>& temp_values)
{
    temp_keys.resize(this->size());
    temp_values.resize(this->size());

    thrust::pair<typename d_vector<system, covariate_key>::iterator,
                 typename d_vector<system, covariate_value>::iterator> out;
    out = thrust::reduce_by_key(this->keys.begin(),
                                this->keys.end(),
                                this->values.begin(),
                                temp_keys.begin(),
                                temp_values.begin(),
                                thrust::equal_to<covariate_key>(),
                                covariate_value_sum<covariate_value>());

    uint32 new_size = out.first - temp_keys.begin();

    this->keys = temp_keys;
    this->values = temp_values;

    this->resize(new_size);
}
METHOD_INSTANTIATE(covariate_observation_table, pack);
METHOD_INSTANTIATE(covariate_empirical_table, pack);

} // namespace firepony
