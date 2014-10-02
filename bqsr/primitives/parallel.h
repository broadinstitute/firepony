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

#include <thrust/scan.h>
#include <thrust/copy.h>

namespace bqsr
{
// simplified version of thrust::inclusive_scan
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline void inclusive_scan(InputIterator first,
                           size_t len,
                           OutputIterator result,
                           Predicate op)
{
    thrust::inclusive_scan(first, first + len, result, op);
}

#include <cub/device/device_select.cuh>

// implementation of copy_if based on CUB
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline size_t copy_if(InputIterator first,
                      size_t len,
                      OutputIterator result,
                      Predicate op,
                      D_VectorU8& temp_storage)
{
    D_Vector<int> num_selected(1);

    // determine amount of temp storage required
    size_t temp_bytes = 0;
    cub::DeviceSelect::If(nullptr,
                          temp_bytes,
                          first,
                          result,
                          num_selected.begin(),
                          len,
                          op);

    // make sure we have enough temp storage
    temp_storage.resize(temp_bytes);

    cub::DeviceSelect::If(thrust::raw_pointer_cast(temp_storage.data()),
                          temp_bytes,
                          first,
                          result,
                          num_selected.begin(),
                          len,
                          op);

    return size_t(num_selected[0]);
}

} // namespace bqsr
