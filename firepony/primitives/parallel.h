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

#ifdef RUN_ON_CPU
#define NO_CUB
#endif

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>

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

#if !defined(NO_CUB)
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
#else
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline size_t copy_if(InputIterator first,
                      size_t len,
                      OutputIterator result,
                      Predicate op,
                      D_VectorU8& temp_storage)
{
    OutputIterator out_last;
    out_last = thrust::copy_if(first, first + len, result, op);
    return out_last - result;
}
#endif

// xxxnsubtil: cub::DeviceSelect::Flagged seems problematic
#if !defined(NO_CUB) && 0
template <typename InputIterator, typename FlagIterator, typename OutputIterator>
inline size_t copy_flagged(InputIterator first,
                           size_t len,
                           OutputIterator result,
                           FlagIterator flags,
                           D_VectorU8& temp_storage)
{
    D_Vector<size_t> num_selected(1);

    // determine amount of temp storage required
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr,
                               temp_bytes,
                               first,
                               flags,
                               result,
                               num_selected.begin(),
                               len);

    // make sure we have enough temp storage
    temp_storage.resize(temp_bytes);

    cub::DeviceSelect::Flagged(thrust::raw_pointer_cast(temp_storage.data()),
                               temp_bytes,
                               first,
                               flags,
                               result,
                               num_selected.begin(),
                               len);

    return size_t(num_selected[0]);
}
#else
struct copy_if_flagged
{
    CUDA_HOST_DEVICE bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

template <typename InputIterator, typename FlagIterator, typename OutputIterator>
inline size_t copy_flagged(InputIterator first,
                           size_t len,
                           OutputIterator result,
                           FlagIterator flags,
                           D_VectorU8& temp_storage)
{
    OutputIterator out_last;
    out_last = thrust::copy_if(first, first + len, flags, result, copy_if_flagged());
    return out_last - result;
}
#endif

#if !defined(NO_CUB)
template <typename InputIterator>
inline int64 sum(InputIterator first,
                 size_t len,
                 D_VectorU8& temp_storage)
{
    D_Vector<int64> result(1);

    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr,
                           temp_bytes,
                           first,
                           result.begin(),
                           len);

    temp_storage.resize(temp_bytes);

    cub::DeviceReduce::Sum(thrust::raw_pointer_cast(temp_storage.data()),
                           temp_bytes,
                           first,
                           result.begin(),
                           len);

    return int64(result[0]);
}
#else
template <typename InputIterator>
inline int64 sum(InputIterator first,
                 size_t len,
                 D_VectorU8& temp_storage)
{
    return thrust::reduce(first, first + len, int64(0));
}
#endif

#if !defined(NO_CUB)
template <typename Key, typename Value>
inline void sort_by_key(D_Vector<Key>& keys,
                        D_Vector<Value>& values,
                        D_Vector<Key>& temp_keys,
                        D_Vector<Value>& temp_values,
                        D_VectorU8& temp_storage,
                        int num_key_bits = sizeof(Key) * 8)
{
    const size_t len = keys.size();
    assert(keys.size() == values.size());

    temp_keys.resize(len);
    temp_values.resize(len);

    cub::DoubleBuffer<Key> d_keys(thrust::raw_pointer_cast(keys.data()),
                                  thrust::raw_pointer_cast(temp_keys.data()));
    cub::DoubleBuffer<Value> d_values(thrust::raw_pointer_cast(values.data()),
                                      thrust::raw_pointer_cast(temp_values.data()));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temp_storage_bytes,
                                    d_keys,
                                    d_values,
                                    len,
                                    0,
                                    num_key_bits);

    temp_storage.resize(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp_storage.data()),
                                    temp_storage_bytes,
                                    d_keys,
                                    d_values,
                                    len,
                                    0,
                                    num_key_bits);

    if (thrust::raw_pointer_cast(keys.data()) != d_keys.Current())
    {
        cudaMemcpy(thrust::raw_pointer_cast(keys.data()), d_keys.Current(), sizeof(Key) * len, cudaMemcpyDeviceToDevice);
    }

    if (thrust::raw_pointer_cast(values.data()) != d_values.Current())
    {
        cudaMemcpy(thrust::raw_pointer_cast(values.data()), d_values.Current(), sizeof(Value) * len, cudaMemcpyDeviceToDevice);
    }
}
#else
template <typename Key, typename Value>
inline void sort_by_key(D_Vector<Key>& keys,
                        D_Vector<Value>& values,
                        D_Vector<Key>& temp_keys,
                        D_Vector<Value>& temp_values,
                        D_VectorU8& temp_storage,
                        int num_key_bits = sizeof(Key) * 8)
{
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
}
#endif

} // namespace bqsr
