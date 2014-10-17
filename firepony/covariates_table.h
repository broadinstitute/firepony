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

typedef uint32 covariate_key;

struct covariate_value
{
    uint32 observations;
    float mismatches;
};

template <typename system_tag>
struct covariate_table_storage
{
    bqsr::vector<system_tag, covariate_key> keys;
    bqsr::vector<system_tag, covariate_value> values;

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

    template <typename other_system_tag>
    void copyfrom(covariate_table_storage<other_system_tag>& other)
    {
        keys = other.keys;
        values = other.values;
    }
};

// host-side covariate tables are just simple storage
typedef covariate_table_storage<host_tag> H_CovariateTable;

// device covariate tables implement merging primitives and views
struct D_CovariateTable : public covariate_table_storage<target_system_tag>
{
    void sort(D_Vector<covariate_key>& temp_keys, D_Vector<covariate_value>& temp_values, D_VectorU8& temp_storage, uint32 num_key_bits);
    void pack(D_Vector<covariate_key>& temp_keys, D_Vector<covariate_value>& temp_values);

    struct view
    {
        D_VectorU32::view keys;
        bqsr::vector<target_system_tag, covariate_value>::view values;

        // note: we don't use implicit cast operators here
        // we want to derive from this view to implement D_CovariatePool::view
        view(D_CovariateTable& table)
            : keys(table.keys),
              values(table.values)
        { }
    };
};
