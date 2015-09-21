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

#include "../types.h"

#include "util.h"
#include "packed_stream.h"

namespace firepony {

// lifted directly from nvbio
// xxxnsubtil: could use some cleanup

template <target_system system, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T = false, typename IndexType = uint32>
struct packed_vector
{
    enum {
        SYMBOL_SIZE         = SYMBOL_SIZE_T,
        IS_BIG_ENDIAN       = BIG_ENDIAN_T,
        WORD_SIZE           = sizeof(uint32) * 8,
        SYMBOLS_PER_WORD    = WORD_SIZE / SYMBOL_SIZE,
    };

    typedef packed_vector<system, SYMBOL_SIZE_T, BIG_ENDIAN_T, IndexType>                type;

    typedef IndexType           index_type;

    typedef typename vector<system, uint32>::pointer                                            pointer;
    typedef typename vector<system, uint32>::const_pointer                                      const_pointer;
    typedef packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN,       pointer, index_type>         stream_type;
    typedef packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN, const_pointer, index_type>         const_stream_type;
    typedef typename stream_type::iterator                                                      iterator;
    typedef typename const_stream_type::iterator                                                const_iterator;
    typedef uint8                                                                               value_type;
    typedef packed_stream_reference<stream_type>                                                reference;
    typedef packed_stream_reference<const_stream_type>                                          const_reference;

    typedef stream_type view;
    typedef const_stream_type const_view;

    packed_vector(const index_type size = 0)
        : m_storage(divide_ri(size, SYMBOLS_PER_WORD)), m_size(size)
    { }

    template <target_system OtherSystemTag>
    packed_vector(const packed_vector<OtherSystemTag, SYMBOL_SIZE, IS_BIG_ENDIAN, IndexType>& other)
        : m_storage(other.m_storage), m_size(other.m_size)
    { }

    void reserve(const index_type size)
    {
        if (m_storage.size() < divide_ri(m_size, SYMBOLS_PER_WORD))
        {
            m_storage.resize(divide_ri(m_size, SYMBOLS_PER_WORD));
        }
    }

    void resize(const index_type size)
    {
        m_size = size;
        reserve(size);
    }

    void clear(void)
    {
        resize(0);
    }

    index_type size() const
    {
        return m_size;
    }

    index_type capacity() const
    {
        return m_storage.size();
    }

    iterator begin()
    {
        return stream_type(m_storage.data(), m_size).begin();
    }

    const_iterator begin() const
    {
        return const_stream_type(m_storage.data(), m_size).begin();
    }

    iterator end()
    {
        return stream_type(m_storage.data(), m_size).begin() + m_size;
    }

    const_iterator end() const
    {
        return const_stream_type(m_storage.data(), m_size).begin() + m_size;
    }

    void push_back(const value_type s)
    {
        if (m_storage.size() < divide_ri(m_size + 1, SYMBOLS_PER_WORD))
        {
            m_storage.resize(divide_ri(m_size + 1, SYMBOLS_PER_WORD));
        }

        begin()[m_size] = s;
        m_size++;
    }

    /// get the i-th symbol
    /// note: no device implementation for this as packed_vectors are never used on device
    /// (only their plain view, which is a packed_stream)
    CUDA_HOST const typename stream_type::symbol_type operator[] (const index_type i) const
    {
        const_stream_type stream(thrust::raw_pointer_cast(&m_storage.front()), m_size);
        return stream[i];
    }

    CUDA_HOST typename stream_type::reference operator[] (const index_type i)
    {
        stream_type stream(thrust::raw_pointer_cast(&m_storage.front()), m_size);
        return stream[i];
    }

    operator view()
    {
        return view(thrust::raw_pointer_cast(m_storage.data()), m_size);
    }

    operator const_view() const
    {
        return const_view(thrust::raw_pointer_cast(m_storage.data()), m_size);
    }

    stream_type stream_at_index(const index_type i)
    {
        return stream_type(thrust::raw_pointer_cast(&m_storage.front()), m_size, i);
    }

    // assignment from a host view
    void copy_from_view(const typename packed_vector<host, SYMBOL_SIZE>::const_view& other)
    {
        m_storage.resize(divide_ri(other.size(), SYMBOLS_PER_WORD));
        m_storage.assign((uint32 *)other.stream(), ((uint32 *)other.stream()) + divide_ri(other.size(), SYMBOLS_PER_WORD));
        m_size = other.size();
    }

    vector<system, uint32> m_storage;
    index_type             m_size;
};

} // namespace firepony
