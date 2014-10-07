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

#include <cub/device/device_radix_sort.cuh>

// xxxnsubtil: move this into primitives/parallel.h
void D_CovariateTable::sort(D_Vector<covariate_key>& temp_keys, D_Vector<covariate_value>& temp_values,
                            D_VectorU8& temp_storage, uint32 num_key_bits)
{
    temp_keys.resize(size());
    temp_values.resize(size());

    cub::DoubleBuffer<covariate_key>   d_keys(thrust::raw_pointer_cast(keys.data()),
                                              thrust::raw_pointer_cast(temp_keys.data()));
    cub::DoubleBuffer<covariate_value> d_values(thrust::raw_pointer_cast(values.data()),
                                                thrust::raw_pointer_cast(temp_values.data()));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temp_storage_bytes,
                                    d_keys,
                                    d_values,
                                    size());

    temp_storage.resize(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp_storage.data()),
                                    temp_storage_bytes,
                                    d_keys,
                                    d_values,
                                    size());

    if (thrust::raw_pointer_cast(keys.data()) != d_keys.Current())
    {
        cudaMemcpy(thrust::raw_pointer_cast(keys.data()), d_keys.Current(), sizeof(covariate_key) * size(), cudaMemcpyDeviceToDevice);
    }

    if (thrust::raw_pointer_cast(values.data()) != d_values.Current())
    {
        cudaMemcpy(thrust::raw_pointer_cast(values.data()), d_values.Current(), sizeof(covariate_value) * size(), cudaMemcpyDeviceToDevice);
    }
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
