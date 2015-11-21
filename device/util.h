/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "../types.h"

namespace firepony {

// implements a pingpong queue between two objects
template <typename V>
struct pingpong_queue
{
    V& a;
    V& b;
    uint8 which;

    pingpong_queue(V& source, V& dest)
        : a(source), b(dest), which(0)
    { }

    V& source(void)
    {
        return (which ? b : a);
    }

    V& dest(void)
    {
        return (which ? a : b);
    }

    void swap(void)
    {
        which ^= 1;
    }

    bool is_swapped(void)
    {
        return (which != 0);
    }
};

template <typename V>
static pingpong_queue<V> make_pingpong_queue(V& source, V& dest)
{
    return pingpong_queue<V>(source, dest);
}

// prepare temp_storage to store num_elements to be packed into a bit vector
template <target_system system> void pack_prepare_storage_2bit(allocation<system, uint8>& storage, uint32 num_elements);
template <target_system system> void pack_prepare_storage_1bit(allocation<system, uint8>& storage, uint32 num_elements);

// packs a vector of uint8 into a bit vector
template <target_system system> void pack_to_2bit(packed_vector<system, 2>& dest, allocation<system, uint8>& src);
template <target_system system> void pack_to_1bit(packed_vector<system, 1>& dest, allocation<system, uint8>& src);

// round a double to the Nth decimal place
double round_n(double val, int n);

} // namespace firepony
