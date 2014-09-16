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

struct D_CovariateTable
{
    D_VectorU32 keys;
    D_VectorU32 observations;

    D_CovariateTable()
    {
        keys.resize(0);
        observations.resize(0);
    }

    void resize(uint32 size)
    {
        keys.resize(size);
        observations.resize(size);
    }

    uint32 size(void)
    {
        assert(keys.size() == observations.size());
        return keys.size();
    }

    void copyfrom(D_CovariateTable& other)
    {
        keys = other.keys;
        observations = other.observations;
    }

    void concatenate(D_CovariateTable& other, uint32 other_size);
    void sort(D_VectorU32& indices, D_VectorU32& temp);
    void pack(D_VectorU32& temp_keys, D_VectorU32& temp_observations);

    struct view
    {
        D_VectorU32::view keys;
        D_VectorU32::view observations;

        view(D_CovariateTable& table)
            : keys(table.keys),
              observations(table.observations)
        { }
    };
};

struct H_CovariateTable
{
    H_VectorU32 keys;
    H_VectorU32 observations;

    H_CovariateTable()
    { };

    void copyfrom(D_CovariateTable& other)
    {
        keys = other.keys;
        observations = other.observations;
    }

    uint32 size(void)
    {
        assert(keys.size() == observations.size());
        return keys.size();
    }
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

    uint32 keys[LOCAL_TABLE_SIZE];
    uint32 observations[LOCAL_TABLE_SIZE];

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

    CUDA_HOST_DEVICE bool insert(uint32 key, uint32 target_index)
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
            observations[0] = 1;
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
                observations[i + 1] = observations[i];
            }
        } else {
            assert(target_index == num_items);
        }

        keys[target_index] = key;
        observations[target_index] = 1;
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
        covariate_table_entry *loc = nvbio::find_pivot(data, num_entries, covariate_key_gequal(key));
        return (int)(loc - data);
#endif
    }
};
