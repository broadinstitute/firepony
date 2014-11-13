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

typedef uint32 covariate_key;

// the value for each row of a covariate observation table
struct covariate_observation_value
{
    uint32 observations;
    float mismatches;
};

// the value for each row of a covariate empirical table
// this includes all the info in the covariate observation table plus computed values
struct covariate_empirical_value
{
    uint32 observations;
    double mismatches;
    double expected_errors;
    double estimated_quality;
    double empirical_quality;
};

// covariate table
// stores a list of key-value pairs, where the key is a covariate_key and the value is either covariate_observation_value or covariate_empirical_value
template <target_system system, typename covariate_value>
struct covariate_table
{
    typedef covariate_value value_type;

    vector<system, covariate_key> keys;
    vector<system, covariate_value> values;

    covariate_table()
    {
        keys.resize(0);
        values.resize(0);
    }

    void resize(size_t size)
    {
        keys.resize(size);
        values.resize(size);
    }

    size_t size(void) const
    {
        assert(keys.size() == values.size());
        return keys.size();
    }

    template <target_system other_system>
    void copyfrom(covariate_table<other_system, covariate_value>& other)
    {
        keys.resize(other.keys.size());
        values.resize(other.values.size());

        thrust::copy(other.keys.begin(), other.keys.end(), keys.begin());
        thrust::copy(other.values.begin(), other.values.end(), values.begin());
    }

    void sort(vector<system, covariate_key>& temp_keys,
              vector<system, covariate_value>& temp_values,
              vector<system, uint8>& temp_storage,
              uint32 num_key_bits);

    void pack(vector<system, covariate_key>& temp_keys,
              vector<system, covariate_value>& temp_values);

    struct view
    {
        typename vector<system, uint32>::view keys;
        typename vector<system, covariate_value>::view values;
    };

    operator view()
    {
        struct view v = {
            keys,
            values,
        };

        return v;
    }
};

template <target_system system> using covariate_observation_table = covariate_table<system, covariate_observation_value>;
template <target_system system> using covariate_empirical_table = covariate_table<system, covariate_empirical_value>;

template <target_system system> void covariate_observation_to_empirical_table(firepony_context<system>& context,
                                                                              const covariate_observation_table<system>& observation_table,
                                                                              covariate_empirical_table<system>& empirical_table);

} // namespace firepony
