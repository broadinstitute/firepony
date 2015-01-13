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

#include "cuda.h"

namespace firepony {

/// find the lower bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value>
inline CUDA_HOST_DEVICE Iterator lower_bound(const Value         x,
                                             Iterator            begin,
                                             const size_t        n)
{
    // if the range has a size of zero, let's just return the intial element
    if (n == 0)
        return begin;

    // check whether this segment is all left or right of x
    if (x < begin[0])
        return begin;

    if (begin[n-1] < x)
        return begin + n;

    // perform a binary search over the given range
    size_t count = n;

    while (count > 0)
    {
        const size_t count2 = count / 2;

        Iterator mid = begin + count2;

        if (*mid < x)
            begin = ++mid, count -= count2 + 1;
        else
            count = count2;
    }
    return begin;
}

} // namespace firepony
