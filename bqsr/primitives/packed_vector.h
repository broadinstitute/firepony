/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
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

#include "util.h"
#include "vector.h"
#include "packed_stream.h"

namespace bqsr
{

// lifted directly from nvbio
// xxxnsubtil: could use some cleanup

template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T = false, typename IndexType = uint32>
struct packed_vector
{
    enum {
        SYMBOL_SIZE         = SYMBOL_SIZE_T,
        IS_BIG_ENDIAN       = BIG_ENDIAN_T,
        SYMBOLS_PER_WORD    = 32 / SYMBOL_SIZE,
    };

    typedef packed_vector<SystemTag, SYMBOL_SIZE_T, BIG_ENDIAN_T, IndexType>                type;

    typedef SystemTag           system_tag;
    typedef IndexType           index_type;

    typedef typename bqsr::vector<system_tag, uint32>::pointer                                  pointer;
    typedef typename bqsr::vector<system_tag, uint32>::const_pointer                            const_pointer;
    typedef bqsr::packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN,       pointer, index_type>   stream_type;
    typedef bqsr::packed_stream<SYMBOL_SIZE, uint8, IS_BIG_ENDIAN, const_pointer, index_type>   const_stream_type;
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

    template <typename OtherSystemTag>
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
        return stream_type(m_storage.data()).begin();
    }

    const_iterator begin() const
    {
        return const_stream_type(m_storage.data()).begin();
    }

    iterator end()
    {
        return stream_type(m_storage.data()).begin() + m_size;
    }

    const_iterator end() const
    {
        return const_stream_type(m_storage.data()).begin() + m_size;
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
        const_stream_type stream(thrust::raw_pointer_cast(&m_storage.front()));
        return stream[i];
    }

    CUDA_HOST typename stream_type::reference operator[] (const index_type i)
    {
        stream_type stream(thrust::raw_pointer_cast(&m_storage.front()));
        return stream[i];
    }

    operator view()
    {
        return view(thrust::raw_pointer_cast(m_storage.data()));
    }

    operator const_view() const
    {
        return const_view(thrust::raw_pointer_cast(m_storage.data()));
    }

    stream_type stream_at_index(const index_type i)
    {
        return stream_type(thrust::raw_pointer_cast(&m_storage.front()), i);
    }

    bqsr::vector<system_tag, uint32> m_storage;
    index_type                       m_size;
};

} // namespace bqsr
