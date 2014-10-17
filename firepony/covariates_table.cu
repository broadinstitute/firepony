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

#include "primitives/parallel.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>

void D_CovariateTable::sort(D_Vector<covariate_key>& temp_keys, D_Vector<covariate_value>& temp_values,
                            D_VectorU8& temp_storage, uint32 num_key_bits)
{
    bqsr::sort_by_key(keys, values, temp_keys, temp_values, temp_storage, num_key_bits);
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
