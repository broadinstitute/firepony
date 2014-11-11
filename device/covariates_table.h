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

struct covariate_value
{
    uint32 observations;
    float mismatches;
};

template <target_system system>
struct covariate_table_storage
{
    vector<system, covariate_key> keys;
    vector<system, covariate_value> values;

    covariate_table_storage()
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
    void copyfrom(covariate_table_storage<other_system>& other)
    {
        keys.resize(other.keys.size());
        values.resize(other.values.size());

        thrust::copy(other.keys.begin(), other.keys.end(), keys.begin());
        thrust::copy(other.values.begin(), other.values.end(), values.begin());
    }
};

// host-side covariate tables are just simple storage
typedef covariate_table_storage<host> h_covariate_table;

// device covariate tables implement merging primitives and views
template <target_system system>
struct d_covariate_table : public covariate_table_storage<system>
{
    void sort(d_vector<system, covariate_key>& temp_keys, d_vector<system, covariate_value>& temp_values, d_vector_u8<system>& temp_storage, uint32 num_key_bits);
    void pack(d_vector<system, covariate_key>& temp_keys, d_vector<system, covariate_value>& temp_values);

    struct view
    {
        typename d_vector<system, uint32>::view keys;
        typename d_vector<system, covariate_value>::view values;

        view(d_covariate_table& table)
            : keys(table.keys),
              values(table.values)
        { }
    };
};

} // namespace firepony
