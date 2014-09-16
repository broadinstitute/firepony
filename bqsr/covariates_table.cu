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


#include "bqsr_types.h"
#include "bqsr_context.h"
#include "covariates.h"
#include "covariates_table.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>

void D_CovariateTable::concatenate(D_CovariateTable& other, uint32 other_size)
{
    uint32 concat_index = size();

    resize(size() + other_size);

    thrust::copy(other.keys.begin(), other.keys.begin() + other_size, keys.begin() + concat_index);
    thrust::copy(other.observations.begin(), other.observations.begin() + other_size, observations.begin() + concat_index);
}

void D_CovariateTable::sort(D_VectorU32& indices, D_VectorU32& temp)
{
    indices.resize(size());
    temp.resize(size());

    thrust::sequence(indices.begin(), indices.end());
    thrust::sort_by_key(keys.begin(), keys.end(), indices.begin());

    thrust::gather(indices.begin(), indices.end(), observations.begin(), temp.begin());
    observations = temp;
}

void D_CovariateTable::pack(D_VectorU32& temp_keys, D_VectorU32& temp_observations)
{
    temp_keys.resize(size());
    temp_observations.resize(size());

    thrust::pair<D_VectorU32::iterator, D_VectorU32::iterator> out;
    out = thrust::reduce_by_key(keys.begin(), keys.end(), observations.begin(), temp_keys.begin(), temp_observations.begin());

    uint32 new_size = out.first - temp_keys.begin();

    keys = temp_keys;
    observations = temp_observations;

    resize(new_size);
}
