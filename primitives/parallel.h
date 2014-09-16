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

// simplified version of thrust::copy_if
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline size_t copy_if(InputIterator first,
                      size_t len,
                      OutputIterator result,
                      Predicate op)
{
    InputIterator last;
    last = thrust::copy_if(first, first + len, result, op);
    return last - result;
}

} // namespace bqsr
