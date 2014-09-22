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
    void concatenate(D_CovariateTable& other, size_t other_size);
    void sort(D_Vector<covariate_key>& indices, D_Vector<covariate_value>& temp);
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


struct D_CovariatePool : public D_CovariateTable
{
    D_VectorU32 items_allocated;

    D_CovariatePool()
    {
        items_allocated.resize(1);
        clear();
    }

    void clear(void)
    {
        items_allocated[0] = 0;
    }

    uint32 allocated(void)
    {
        return items_allocated[0];
    }

    struct view : public D_CovariateTable::view
    {
        D_VectorU32::view items_allocated;

        view(D_CovariatePool& pool)
            : D_CovariateTable::view(pool),
              items_allocated(pool.items_allocated)
        { }

        CUDA_DEVICE uint32 allocate(const uint32 num_items)
        {
            uint32 old;

            old = atomicAdd(&items_allocated[0], num_items);
            if (old + num_items > keys.size())
            {
                atomicSub(&items_allocated[0], num_items);
                return uint32(-1);
            }

            return old;
        }
    };
};

#define LOCAL_TABLE_SIZE 256

struct D_LocalCovariateTable
{
    enum {
        max_size = LOCAL_TABLE_SIZE
    };

    covariate_key keys[LOCAL_TABLE_SIZE];
    covariate_value values[LOCAL_TABLE_SIZE];

    uint32 num_items;

    CUDA_HOST_DEVICE D_LocalCovariateTable()
        : num_items(0)
    { }

    CUDA_HOST_DEVICE uint32 size(void) const
    {
        return num_items;
    }

    CUDA_HOST_DEVICE bool exists(uint32 index)
    {
        if (index < num_items)
            return true;

        return false;
    }

    CUDA_HOST_DEVICE bool insert(uint32 key, uint32 target_index, float mismatch)
    {
        if (num_items == LOCAL_TABLE_SIZE - 1)
        {
            // overflow
            return false;
        }

        if (num_items == 0)
        {
            // simple case
            assert(target_index == 0);
            keys[0] = key;
            values[0] = { 1, mismatch }; // observations, mismatches
            num_items = 1;
            return true;
        }

        if (target_index < num_items)
        {
            // move all entries from target_index to num_entries forward
            // xxxnsubtil: fix signed vs. unsigned issues
            for(int i = int(num_items) - 1; i >= int(target_index); i--)
            {
                keys[i + 1] = keys[i];
                values[i + 1] = values[i];
            }
        } else {
            assert(target_index == num_items);
        }

        keys[target_index] = key;
        values[target_index] = { 1, mismatch };
        num_items++;

        return true;
    }

    CUDA_HOST_DEVICE int find_insertion_point(uint32 key)
    {
        uint32 i;

        if (num_items == 0)
            return 0;

#if 1
        for(i = 0; i < num_items; i++)
        {
            if (keys[i] >= key)
                break;
        }

        return i;
#else
        covariate_value *loc = nvbio::find_pivot(data, num_entries, covariate_key_gequal(key));
        return (int)(loc - data);
#endif
    }
};
