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
template <target_system system> void pack_prepare_storage_2bit(d_vector_u8<system>& storage, uint32 num_elements);
template <target_system system> void pack_prepare_storage_1bit(d_vector_u8<system>& storage, uint32 num_elements);

// packs a vector of uint8 into a bit vector
template <target_system system> void pack_to_2bit(d_packed_vector_2b<system>& dest, d_vector_u8<system>& src);
template <target_system system> void pack_to_1bit(d_packed_vector_1b<system>& dest, d_vector_u8<system>& src);

// round a double to the Nth decimal place
double round_n(double val, int n);

} // namespace firepony
