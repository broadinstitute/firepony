/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#define WAR_CUB_COPY_FLAGGED 1

#include "../../types.h"
#include "backends.h"

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
// silence warnings from debug code in cub
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <cub/device/device_radix_sort.cuh>
#pragma GCC diagnostic pop

namespace firepony {

struct copy_if_flagged
{
    CUDA_HOST_DEVICE bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

// thrust-based implementation of parallel primitives
template <target_system system>
struct parallel_thrust
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
        return thrust::for_each(firepony::backend_policy<system>::execution_policy(), first, last, f);
    }

    // shortcut to run for_each on a whole vector
    template <typename T, typename UnaryFunction>
    static inline typename vector<system, T>::iterator for_each(vector<system, T>& vector, UnaryFunction f)
    {
        return thrust::for_each(firepony::backend_policy<system>::par, vector.begin(), vector.end(), f);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 d_vector_u8<system>& temp_storage)
    {
        // use the fallback thrust version
        OutputIterator out_last;
        out_last = thrust::copy_if(first, first + len, result, op);
        return out_last - result;
    }

    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      d_vector_u8<system>& temp_storage)
    {
        OutputIterator out_last;
        out_last = thrust::copy_if(first, first + len, flags, result, copy_if_flagged());
        return out_last - result;
    }

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            d_vector_u8<system>& temp_storage)
    {
        return thrust::reduce(first, first + len, int64(0));
    }

    template <typename Key, typename Value>
    static inline void sort_by_key(d_vector<system, Key>& keys,
                                   d_vector<system, Value>& values,
                                   d_vector<system, Key>& temp_keys,
                                   d_vector<system, Value>& temp_values,
                                   d_vector_u8<system>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8)
    {
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    }

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(vector<system, Key>& keys,
                                       vector<system, Value>& values,
                                       vector<system, Key>& output_keys,
                                       vector<system, Value>& output_values,
                                       vector<system, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        auto out = thrust::reduce_by_key(keys.begin(),
                                         keys.end(),
                                         values.begin(),
                                         output_keys.begin(),
                                         output_values.begin(),
                                         thrust::equal_to<Key>(),
                                         reduction_op);
        return out.first - output_keys.begin();
    }

    static inline void synchronize()
    { }
};

// default to thrust
template <target_system system>
struct parallel : public parallel_thrust<system>
{
    using parallel_thrust<system>::for_each;
    using parallel_thrust<system>::inclusive_scan;
    using parallel_thrust<system>::copy_if;
    using parallel_thrust<system>::copy_flagged;
    using parallel_thrust<system>::sum;
    using parallel_thrust<system>::sort_by_key;
    using parallel_thrust<system>::synchronize;
    using parallel_thrust<system>::reduce_by_key;
};

#if ENABLE_CUDA_BACKEND
// specialization for the cuda backend based on CUB primitives
template <>
struct parallel<cuda> : public parallel_thrust<cuda>
{
    using parallel_thrust<cuda>::for_each;

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 d_vector_u8<cuda>& temp_storage)
    {
        d_vector_i32<cuda> num_selected(1);

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

    // xxxnsubtil: cub::DeviceSelect::Flagged seems problematic
#if !WAR_CUB_COPY_FLAGGED
    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      d_vector_u8<cuda>& temp_storage)
    {
        d_vector<cuda, size_t> num_selected(1);

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
    using parallel_thrust<cuda>::copy_flagged;
#endif

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            d_vector_u8<cuda>& temp_storage)
    {
        d_vector_i64<cuda> result(1);

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

    template <typename Key, typename Value>
    static inline void sort_by_key(d_vector<cuda, Key>& keys,
                                   d_vector<cuda, Value>& values,
                                   d_vector<cuda, Key>& temp_keys,
                                   d_vector<cuda, Value>& temp_values,
                                   d_vector_u8<cuda>& temp_storage,
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

    // returns the size of the output key/value vectors
    template <typename Key, typename Value, typename ReductionOp>
    static inline size_t reduce_by_key(vector<cuda, Key>& keys,
                                       vector<cuda, Value>& values,
                                       vector<cuda, Key>& output_keys,
                                       vector<cuda, Value>& output_values,
                                       vector<cuda, uint8>& temp_storage,
                                       ReductionOp reduction_op)
    {
        const size_t len = keys.size();
        assert(keys.size() == values.size());

        output_keys.resize(len);
        output_values.resize(len);

        vector<cuda, uint32> num_segments(1);

        size_t temp_storage_bytes = 0;

        cub::DeviceReduce::ReduceByKey(nullptr,
                                       temp_storage_bytes,
                                       thrust::raw_pointer_cast(keys.data()),
                                       thrust::raw_pointer_cast(output_keys.data()),
                                       thrust::raw_pointer_cast(values.data()),
                                       thrust::raw_pointer_cast(output_values.data()),
                                       thrust::raw_pointer_cast(num_segments.data()),
                                       reduction_op,
                                       keys.size());

        temp_storage.resize(temp_storage_bytes);

        cub::DeviceReduce::ReduceByKey(thrust::raw_pointer_cast(temp_storage.data()),
                                       temp_storage_bytes,
                                       thrust::raw_pointer_cast(keys.data()),
                                       thrust::raw_pointer_cast(output_keys.data()),
                                       thrust::raw_pointer_cast(values.data()),
                                       thrust::raw_pointer_cast(output_values.data()),
                                       thrust::raw_pointer_cast(num_segments.data()),
                                       reduction_op,
                                       keys.size());

        return num_segments[0];
    }

    static inline void synchronize(void)
    {
        cudaDeviceSynchronize();
    }
};
#endif

} // namespace firepony

