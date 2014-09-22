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

void D_CovariateTable::concatenate(D_CovariateTable& other, size_t other_size)
{
    size_t concat_index = size();

    resize(size() + other_size);

    thrust::copy(other.keys.begin(), other.keys.begin() + other_size, keys.begin() + concat_index);
    thrust::copy(other.values.begin(), other.values.begin() + other_size, values.begin() + concat_index);
}

void D_CovariateTable::sort(D_VectorU32& indices, D_Vector<covariate_value>& temp)
{
    indices.resize(size());
    temp.resize(size());

    thrust::sequence(indices.begin(), indices.end());
    thrust::sort_by_key(keys.begin(), keys.end(), indices.begin());

    thrust::gather(indices.begin(), indices.end(), values.begin(), temp.begin());
    values = temp;
}

struct covariate_value_sum
{
    CUDA_HOST_DEVICE covariate_value operator() (const covariate_value& a, const covariate_value& b)
    {
        return covariate_value { a.observations + b.observations,
                                 a.mismatches + b.mismatches };
    }
};


void D_CovariateTable::pack(D_Vector<covariate_key>& temp_keys, D_Vector<covariate_value>& temp_values)
{
    temp_keys.resize(size());
    temp_values.resize(size());

    thrust::pair<D_Vector<covariate_key>::iterator, D_Vector<covariate_value>::iterator> out;
    out = thrust::reduce_by_key(keys.begin(),
                                keys.end(),
                                values.begin(),
                                temp_keys.begin(),
                                temp_values.begin(),
                                thrust::equal_to<covariate_key>(),
                                covariate_value_sum());

    uint32 new_size = out.first - temp_keys.begin();

    keys = temp_keys;
    values = temp_values;

    resize(new_size);
}
