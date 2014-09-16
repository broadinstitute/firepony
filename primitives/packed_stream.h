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

#include "cuda.h"
#include "util.h"
#include "packed_stream_packer.h"

#include <thrust/iterator/iterator_traits.h>

namespace bqsr
{

// lifted directly from nvbio
// xxxnsubtil: could use some cleanup

template <typename Stream>
struct packed_stream_reference
{
    typedef Stream stream_type;
    typedef typename Stream::symbol_type symbol_type;

    CUDA_HOST_DEVICE packed_stream_reference(stream_type stream)
        : m_stream(stream)
    { }

    CUDA_HOST_DEVICE packed_stream_reference& operator=(const packed_stream_reference& ref)
    {
        return (*this = symbol_type(ref));
    }

    CUDA_HOST_DEVICE packed_stream_reference& operator=(const symbol_type s)
    {
        m_stream.set(0u, s);
        return *this;
    }

    CUDA_HOST_DEVICE operator symbol_type() const
    {
        return m_stream.get(0u);
    }

    stream_type m_stream;
};

template <uint32 SYMBOL_SIZE_T, typename Symbol, bool BIG_ENDIAN_T, typename InputStream, typename IndexType>
struct packed_stream
{
    typedef packed_stream<SYMBOL_SIZE_T, Symbol, BIG_ENDIAN_T, InputStream, IndexType> type;

    enum {
        SYMBOL_SIZE             = SYMBOL_SIZE_T,
        SYMBOL_COUNT            = 1u << SYMBOL_SIZE,
        SYMBOL_MASK             = SYMBOL_COUNT - 1u,
        IS_BIG_ENDIAN           = BIG_ENDIAN_T,
        ALPHABET_SIZE           = SYMBOL_COUNT,
    };

    typedef IndexType                                                   index_type;
    typedef typename signed_type<IndexType>::type                       sindex_type;
    typedef Symbol                                                      symbol_type;
    typedef InputStream                                                 stream_type;
    typedef type                                                        iterator;
    typedef typename thrust::iterator_traits<stream_type>::value_type   storage_type;
    typedef packed_stream_reference<type>                               reference;
    typedef symbol_type                                                 const_reference;

    typedef typename thrust::iterator_traits<stream_type>::iterator_category    iterator_category;
    typedef symbol_type                                                         value_type;
    typedef typename signed_type<index_type>::type                              difference_type;
    typedef typename signed_type<index_type>::type                              distance_type;

    CUDA_HOST_DEVICE packed_stream() = default;
//    CUDA_HOST_DEVICE packed_stream(const stream_type stream, const index_type index = 0)
//        : m_stream(stream), m_index(index)
//    { }

    template <typename UInputStream>
    CUDA_HOST_DEVICE explicit packed_stream(const UInputStream stream, const index_type index = 0)
        : m_stream( static_cast<InputStream>(stream) ), m_index( index )
    { }

    CUDA_HOST_DEVICE reference operator*() const
    {
        return reference(*this);
    }

    CUDA_HOST_DEVICE symbol_type operator[](const index_type i) const
    {
        return get(i);
    }

    CUDA_HOST_DEVICE reference operator[](const index_type i)
    {
        return reference(*this + i);
    }

    CUDA_HOST_DEVICE symbol_type get(const index_type i) const
    {
        return packer<IS_BIG_ENDIAN,
                      SYMBOL_SIZE,
                      symbol_type,
                      stream_type,
                      index_type,
                      storage_type>::get_symbol(m_stream, i + m_index);
    }

    CUDA_HOST_DEVICE void set(const index_type i, const symbol_type s)
    {
        return packer<IS_BIG_ENDIAN,
                      SYMBOL_SIZE,
                      symbol_type,
                      stream_type,
                      index_type,
                      storage_type>::set_symbol(m_stream, i + m_index, s);
    }

    CUDA_HOST_DEVICE iterator begin() const
    {
        return iterator(m_stream, m_index);
    }

    CUDA_HOST_DEVICE stream_type stream() const
    {
        return m_stream;
    }

    CUDA_HOST_DEVICE index_type index() const
    {
        return m_index;
    }

    CUDA_HOST_DEVICE packed_stream& operator++ ()
    {
        m_index++;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream& operator++ (int)
    {
        m_index++;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream& operator-- ()
    {
        m_index--;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream& operator-- (int)
    {
        m_index--;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream& operator+= (const sindex_type distance)
    {
        m_index += distance;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream& operator-= (const sindex_type distance)
    {
        m_index -= distance;
        return *this;
    }

    CUDA_HOST_DEVICE packed_stream operator+ (const sindex_type distance) const
    {
        return type(m_stream, m_index + distance);
    }

    CUDA_HOST_DEVICE packed_stream operator- (const sindex_type distance) const
    {
        return type(m_stream, m_index - distance);
    }

    CUDA_HOST_DEVICE sindex_type operator- (const packed_stream it) const
    {
        return sindex_type(m_index - it.m_index);
    }

private:
    stream_type m_stream;
    index_type m_index;
};

} // namespace bqsr
