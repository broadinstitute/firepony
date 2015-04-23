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

#include "types.h"

namespace firepony {

struct serialization
{
    template <typename T>
    static inline size_t serialized_size(const T&);
    template <typename T>
    static inline size_t serialized_size(const vector<host, T>&);
    template <typename T>
    static inline size_t serialized_size(const std::vector<T>&);
    template <uint32 bits>
    static inline size_t serialized_size(const packed_vector<host, bits>&);

    template <typename T>
    static inline void *serialize(void *out, const T& in);
    template <typename T>
    static inline void *serialize(void *out, const vector<host, T>&);
    template <typename T>
    static inline void *serialize(void *out, const std::vector<T>&);
    template <uint32 bits>
    static inline void *serialize(void *out, const packed_vector<host, bits>&);

    template <typename T>
    static inline void *unserialize(T *out, void *in);
    template <typename T>
    static inline void *unserialize(vector<host, T> *out, void *in);
    template <typename T>
    static inline void *unserialize(std::vector<T> *out, void *in);
    template <uint32 bits>
    static inline void *unserialize(packed_vector<host, bits> *out, void *in);
};

} // namespace firepony

#include "serialization_inl.h"

