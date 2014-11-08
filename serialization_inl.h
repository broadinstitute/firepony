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

namespace firepony {

// serialization primitives
namespace serialization {

template <typename T>
inline void *decode(T *out, void *in, size_t n_elem)
{
    memcpy(out, in, sizeof(T) * n_elem);
    return static_cast<char *>(in) + sizeof(T) * n_elem;
}

template <>
inline void *decode(std::string *out, void *in, size_t n_elem)
{
    for(size_t i = 0; i < n_elem; i++)
    {
        out[i] = std::string((char *)in);
        in = static_cast<char *>(in) + out[i].size() + 1;
    }

    return in;
}

template <typename T>
inline void *decode(std::vector<T> *out, void *in)
{
    uint64 size;
    in = decode(&size, in);

    out->resize(size);
    in = decode(out->data(), in, size);

    return in;
}

template <typename T>
inline void *decode(vector<host, T> *out, void *in)
{
    uint64 size;
    in = decode(&size, in);

    out->resize(size);
    in = decode(out->data(), in, size);

    return in;
}

template <uint32 bits>
inline void *decode(packed_vector<host, bits> *out, void *in)
{
    in = decode(&out->m_size, in);
    in = decode(&out->m_storage, in);

    return in;
}

template <typename T>
inline void *encode(void *out, const T *in, size_t n_elem)
{
    memcpy(out, in, sizeof(T) * n_elem);
    return static_cast<char *>(out) + sizeof(T) * n_elem;
}

template <>
inline void *encode(void *out, const std::string *in, size_t n_elem)
{
    for(size_t i = 0; i < n_elem; i++)
    {
        out = encode(out, in[i].c_str(), in[i].size() + 1);
    }

    return out;
}

template <typename T>
inline void *encode(void *out, const std::vector<T> *in)
{
    uint64 size = in->size();
    out = encode(out, &size);

    for(uint64 i = 0; i < size; i++)
    {
        out = encode(out, &(*in)[i]);
    }

    return out;
}

template <typename T>
inline void *encode(void *out, const vector<host, T> *in)
{
    uint64 size = in->size();
    out = encode(out, &size);

    for(uint64 i = 0; i < size; i++)
    {
        out = encode(out, &(*in)[i]);
    }

    return out;
}

template <uint32 bits>
inline void *encode(void *out, packed_vector<host, bits> *in)
{
    out = encode(out, &in->m_size);
    out = encode(out, &in->m_storage);

    return out;
}

template <typename T>
inline size_t serialized_size(const T&)
{
    return sizeof(T);
}

template <>
inline size_t serialized_size(const std::string& str)
{
    return str.size() + 1;
}

template <typename T>
inline size_t serialized_size(const std::vector<T>& vec)
{
    return sizeof(uint64) + sizeof(T) * vec.size();
}

template <>
inline size_t serialized_size(const std::vector<std::string>& vec)
{
    size_t ret = sizeof(uint64);

    for(const auto& str : vec)
    {
        ret += str.size() + 1;
    }

    return ret;
}

template <typename T>
inline size_t serialized_size(const vector<host, T>& vec)
{
    return sizeof(uint64) + sizeof(T) * vec.size();
}

//template <typename system_tag, uint32 bits>
//inline size_t serialized_size(const packed_vector<system_tag, bits>& vec)
//{
//    return serialized_size(vec.m_size) + serialized_size(vec.m_storage);
//}

template <typename T>
inline void *unwrap_vector_view(T& view, void *in)
{
    uint64 size;

    in = serialization::decode(&size, in);

    view = T(typename T::iterator(in), typename T::size_type(size));
    in = (void *) (static_cast<typename T::pointer>(in) + size);

    return in;
}

template <typename T>
inline void *unwrap_packed_vector_view(T& view, void *in)
{
    uint64 temp;
    uint32 m_size;
    in = serialization::decode(&m_size, in);
    in = serialization::decode(&temp, in);

    assert(temp == divide_ri(m_size, T::SYMBOLS_PER_WORD));

    view = T(in, m_size);
    in = static_cast<uint32*>(in) + divide_ri(m_size, T::SYMBOLS_PER_WORD);

    return in;
}

} // namespace serialization

} // namespace firepony
