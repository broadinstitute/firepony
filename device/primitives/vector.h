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

#include "backends.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/reverse_iterator.h>

namespace firepony {

template <typename T, typename IndexType = uint64>
struct vector_view
{
    // note: vector_view is *not* a container, it wraps a pointer to T

    // vector_view implements a subset of the std::vector interface that can be used on both host and device
    // the following types/methods from std::vector are *not* present
    //
    // allocator_type
    // operator=()
    // resize()
    // reserve()
    // shrink_to_fit()
    // assign()
    // push_back()
    // pop_back()
    // insert()
    // erase()
    // swap()
    // clear()
    // emplace()
    // emplace_back()
    // get_allocator()

    typedef T*                                                          iterator;
    typedef const T*                                                    const_iterator;
    typedef typename thrust::iterator_traits<iterator>::value_type      value_type;
    typedef typename thrust::iterator_traits<iterator>::reference       reference;
    typedef typename thrust::iterator_traits<const_iterator>::reference const_reference;
    typedef typename thrust::iterator_traits<iterator>::pointer         pointer;
    typedef typename thrust::iterator_traits<const_iterator>::pointer   const_pointer;
    typedef typename thrust::reverse_iterator<iterator>                 reverse_iterator;
    typedef typename thrust::reverse_iterator<const_iterator>           const_reverse_iterator;
    typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
    typedef IndexType                                                   size_type;

    CUDA_HOST_DEVICE vector_view()
        : m_vec(nullptr), m_size(0)
    { }

    CUDA_HOST_DEVICE vector_view(iterator vec, size_type size)
        : m_vec(vec), m_size(size)
    { }

    CUDA_HOST_DEVICE iterator begin()
    {
        return iterator(m_vec);
    }

    CUDA_HOST_DEVICE const_iterator begin() const
    {
        return const_iterator(m_vec);
    }

    CUDA_HOST_DEVICE const_iterator cbegin() const
    {
        return const_iterator(m_vec);
    }

    CUDA_HOST_DEVICE iterator end()
    {
        return iterator(m_vec + m_size);
    }

    CUDA_HOST_DEVICE const_iterator end() const
    {
        return const_iterator(m_vec + m_size);
    }


    CUDA_HOST_DEVICE const_iterator cend() const
    {
        return const_iterator(m_vec + m_size);
    }

    CUDA_HOST_DEVICE reverse_iterator rbegin()
    {
        return reverse_iterator(end());
    }

    CUDA_HOST_DEVICE const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(end());
    }

    CUDA_HOST_DEVICE const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator(end());
    }

    CUDA_HOST_DEVICE reverse_iterator rend()
    {
        return reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE const_reverse_iterator rend() const
    {
        return const_reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE const_reverse_iterator crend() const
    {
        return const_reverse_iterator(begin());
    }

    CUDA_HOST_DEVICE size_type size() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE size_type max_size() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE size_type capacity() const
    {
        return m_size;
    }

    CUDA_HOST_DEVICE bool empty() const
    {
        return m_size == 0;
    }

    CUDA_HOST_DEVICE reference operator[](size_type n)
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE const_reference operator[](size_type n) const
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE reference at(size_type n)
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE const_reference at(size_type n) const
    {
        return m_vec[n];
    }

    CUDA_HOST_DEVICE reference front()
    {
        return m_vec[0];
    }

    CUDA_HOST_DEVICE const_reference front() const
    {
        return m_vec[0];
    }

    CUDA_HOST_DEVICE reference back()
    {
        return m_vec[m_size - 1];
    }

    CUDA_HOST_DEVICE const_reference back() const
    {
        return m_vec[m_size - 1];
    }

    CUDA_HOST_DEVICE value_type* data() noexcept
    {
        return m_vec;
    }

    CUDA_HOST_DEVICE const value_type* data() const noexcept
    {
        return m_vec;
    }

private:
    size_type m_size;
    iterator m_vec;
};

template <target_system system, typename T>
struct vector : public backend_vector_type<system, T>::vector_type
{
    typedef typename backend_vector_type<system, T>::vector_type   base;
    using base::base;   // inherit constructors from the base vector

    typedef vector_view<T> view;
    typedef vector_view<const T> const_view;

    operator view()
    {
        return view(base::size() ? thrust::raw_pointer_cast(base::data()) : nullptr,
                    base::size());
    }

    operator const_view() const
    {
        return const_view(base::size() ? thrust::raw_pointer_cast(base::data()) : nullptr,
                          base::size());
    }

    // assignment from a host vector view
    void copy_from_view(const typename vector<host, T>::const_view& other)
    {
        base::assign(other.begin(), other.end());
    }
};

// prevent std::string vectors from being created on the device
template<target_system system>
struct vector<system, std::string>
{
    vector() = delete;
};

template<>
struct vector<host, std::string> : public std::vector<std::string>
{
    typedef std::vector<std::string> base;
    using base::base;
};

} // namespace firepony
