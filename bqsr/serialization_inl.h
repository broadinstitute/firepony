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
inline void *decode(bqsr::vector<host_tag, T> *out, void *in)
{
    uint64 size;
    in = decode(&size, in);

    out->resize(size);
    in = decode(out->data(), in, size);

    return in;
}

template <uint32 bits>
inline void *decode(bqsr::packed_vector<host_tag, bits> *out, void *in)
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
inline void *encode(void *out, const bqsr::vector<host_tag, T> *in)
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
inline void *encode(void *out, bqsr::packed_vector<host_tag, bits> *in)
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
inline size_t serialized_size(const bqsr::vector<host_tag, T>& vec)
{
    return sizeof(uint64) + sizeof(T) * vec.size();
}

//template <typename system_tag, uint32 bits>
//inline size_t serialized_size(const bqsr::packed_vector<system_tag, bits>& vec)
//{
//    return serialized_size(vec.m_size) + serialized_size(vec.m_storage);
//}

} // namespace serialization