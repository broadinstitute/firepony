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

#include "../../types.h"

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

    typedef typename persistent_allocation<system, uint32>::pointer_type                        pointer;
    typedef typename persistent_allocation<system, uint32>::const_pointer_type                  const_pointer;
    typedef packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN,       pointer, index_type>         stream_type;
    typedef packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN, const_pointer, index_type>         const_stream_type;
    typedef typename stream_type::iterator                                                      iterator;
    typedef typename const_stream_type::iterator                                                const_iterator;
    typedef uint8                                                                               value_type;
    typedef packed_stream_reference<stream_type>                                                reference;
    typedef packed_stream_reference<const_stream_type>                                          const_reference;

    packed_vector()
        : m_storage(), m_size(0)
    { }

    packed_vector(const index_type size)
        : m_storage(divide_ri(size, SYMBOLS_PER_WORD)), m_size(size)
    { }

    template <target_system OtherSystemTag>
    packed_vector(const packed_vector<OtherSystemTag, SYMBOL_SIZE, IS_BIG_ENDIAN, IndexType>& other)
        : m_storage(), m_size(other.m_size)
    {
        m_storage.copy(other.m_storage);
    }

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

    LIFT_HOST_DEVICE index_type size() const
    {
        return m_size;
    }

    LIFT_HOST_DEVICE index_type capacity() const
    {
        return m_storage.size();
    }

    LIFT_HOST_DEVICE iterator begin()
    {
        return stream_type(m_storage.data(), m_size).begin();
    }

    LIFT_HOST_DEVICE const_iterator begin() const
    {
        return const_stream_type(m_storage.data(), m_size).begin();
    }

    LIFT_HOST_DEVICE iterator end()
    {
        return stream_type(m_storage.data(), m_size).begin() + m_size;
    }

    LIFT_HOST_DEVICE const_iterator end() const
    {
        return const_stream_type(m_storage.data(), m_size).begin() + m_size;
    }

    void push_back(const value_type s)
    {
        if (m_storage.size() < divide_ri(m_size + 1, SYMBOLS_PER_WORD))
        {
            m_storage.resize(divide_ri(m_size + 1, SYMBOLS_PER_WORD) * 2);
        }

        begin()[m_size] = s;
        m_size++;
    }

    /// get the i-th symbol
    LIFT_HOST_DEVICE const typename stream_type::symbol_type operator[] (const index_type i) const
    {
        const_stream_type stream(m_storage.data(), m_size);
        return stream[i];
    }

    LIFT_HOST_DEVICE typename stream_type::reference operator[] (const index_type i)
    {
        stream_type stream(m_storage.data(), m_size);
        return stream[i];
    }

    LIFT_HOST_DEVICE stream_type stream_at_index(const index_type i)
    {
        return stream_type(m_storage.data(), m_size, i);
    }

    LIFT_HOST_DEVICE const_stream_type stream_at_index(const index_type i) const
    {
        return const_stream_type(m_storage.data(), m_size, i);
    }

    LIFT_HOST_DEVICE stream_type stream(void)
    {
        return stream_type(m_storage.data(), m_size, 0);
    }

    LIFT_HOST_DEVICE const_stream_type stream(void) const
    {
        return const_stream_type(m_storage.data(), m_size, 0);
    }

    template <target_system other_system>
    void copy(const packed_vector<other_system, SYMBOL_SIZE_T, BIG_ENDIAN_T, IndexType>& other)
    {
        m_storage.copy(other.m_storage);
        m_size = other.size();
    }

    void free(void)
    {
        m_storage.free();
        m_size = 0;
    }

    persistent_allocation<system, uint32> m_storage;
    index_type             m_size;
};

} // namespace firepony
