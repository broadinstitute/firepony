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

namespace serialization {

    template <typename T> void *decode(T *output, void *input, size_t n_elem = 1);
    template <> void *decode(std::string *output, void *input, size_t n_elem);
    template <typename T> void *decode(std::vector<T> *output, void *input);
    template <typename T> void *decode(vector<host, T> *output, void *input);
    template <uint32 bits> void *decode(packed_vector<host, bits> *output, void *input);

    template <typename T> void *encode(void *output, const T *input, size_t n_elem = 1);
    template <> void *encode(void *output, const std::string *input, size_t n_elem);
    template <typename T> void *encode(void *output, const std::vector<T> *input);
    template <typename T> void *encode(void *output, const vector<host, T> *input);
    template <uint32 bits> void *encode(void *output, packed_vector<host, bits> *input);

    template <typename T> size_t serialized_size(const T&);
    template <> size_t serialized_size(const std::string& str);
    template <typename T> size_t serialized_size(const std::vector<T>& vec);
    template <> size_t serialized_size(const std::vector<std::string>& vec);
    template <typename T> size_t serialized_size(const vector<host, T>& vec);

    template <typename T> void *unwrap_vector_from_view(T& view, void *in);
    template <typename T> void *unwrap_packed_vector_view(T& view, void *in);

} // namespace serialization

} // namespace firepony

#include "serialization_inl.h"
