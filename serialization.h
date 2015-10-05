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

#include "types.h"

namespace firepony {

struct serialization
{
    template <typename T>
    static inline size_t serialized_size(const T&);
    template <typename T>
    static inline size_t serialized_size(const vector<host, T>&);
    template <typename T>
    static inline size_t serialized_size(const allocation<host, T>&);
    template <typename T>
    static inline size_t serialized_size(const persistent_allocation<host, T>&);
    template <typename T>
    static inline size_t serialized_size(const std::vector<T>&);
    template <uint32 bits>
    static inline size_t serialized_size(const packed_vector<host, bits>&);

    template <typename T>
    static inline void *serialize(void *out, const T& in);
    template <typename T>
    static inline void *serialize(void *out, const vector<host, T>&);
    template <typename T>
    static inline void *serialize(void *out, const allocation<host, T>&);
    template <typename T>
    static inline void *serialize(void *out, const persistent_allocation<host, T>&);
    template <typename T>
    static inline void *serialize(void *out, const std::vector<T>&);
    template <uint32 bits>
    static inline void *serialize(void *out, const packed_vector<host, bits>&);

    template <typename T>
    static inline void *unserialize(T *out, void *in);
    template <typename T>
    static inline void *unserialize(vector<host, T> *out, void *in);
    template <typename T>
    static inline void *unserialize(allocation<host, T> *out, void *in);
    template <typename T>
    static inline void *unserialize(persistent_allocation<host, T> *out, void *in);
    template <typename T>
    static inline void *unserialize(std::vector<T> *out, void *in);
    template <uint32 bits>
    static inline void *unserialize(packed_vector<host, bits> *out, void *in);
};

} // namespace firepony

#include "serialization_inl.h"

