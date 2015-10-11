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

#include "variant_database.h"

namespace firepony {

// serialization primitives

//// arbitrary integral type
template <typename T>
inline size_t serialization::serialized_size(const T&)
{
    return sizeof(T);
}

template <typename T>
inline void *serialization::serialize(void *out, const T& in)
{
    T data = in;
    memcpy(out, &data, sizeof(T));
    return static_cast<char *>(out) + sizeof(T);
}

template <typename T>
inline void *serialization::unserialize(T *out, void *in)
{
    memcpy(out, in, sizeof(T));
    return static_cast<char *>(in) + sizeof(T);
}


//// std::string
template <>
inline size_t serialization::serialized_size<std::string>(const std::string& str)
{
    return str.size() + 1;
}

template <>
inline void *serialization::serialize<std::string>(void *out, const std::string& in)
{
    size_t len = strlen(in.c_str()) + 1;
    memcpy(out, in.c_str(), len);
    out = static_cast<char *>(out) + len;

    return out;
}

template <>
inline void *serialization::unserialize<std::string>(std::string *out, void *in)
{
    *out = std::string((char *)in);
    in = static_cast<char *>(in) + out->size() + 1;

    return in;
}


//// std::vector
template <typename T>
inline size_t serialization::serialized_size(const std::vector<T>& vec)
{
    return sizeof(size_t) + sizeof(T) * vec.size();
}

template <typename T>
inline void *serialization::serialize(void *out, const std::vector<T>& in)
{
    size_t size = in.size();
    out = serialize(out, size);

    for(size_t i = 0; i < size; i++)
    {
        out = serialize(out, in[i]);
    }

    return out;
}

template <typename T>
inline void *serialization::unserialize(std::vector<T> *out, void *in)
{
    size_t size;
    in = unserialize(&size, in);

    out->resize(size);
    for(size_t i = 0; i < size; i++)
    {
        in = unserialize(&((*out)[i]), in);
    }

    return in;
}


//// lift allocation
template <typename T>
inline size_t serialization::serialized_size(const allocation<host, T>& vec)
{
    return sizeof(size_t) + sizeof(T) * vec.size();
}

template <typename T>
inline void *serialization::serialize(void *out, const allocation<host, T>& in)
{
    size_t size = in.size();
    out = serialize(out, size);

    for(size_t i = 0; i < size; i++)
    {
        out = serialize(out, in[i]);
    }

    return out;
}

template <typename T>
inline void *serialization::unserialize(allocation<host, T> *out, void *in)
{
    size_t size;
    in = unserialize(&size, in);

    out->resize(size);

    for(size_t i = 0; i < size; i++)
    {
        in = unserialize(&((*out)[i]), in);
    }

    return in;
}


//// lift persistent allocation
template <typename T>
inline size_t serialization::serialized_size(const persistent_allocation<host, T>& vec)
{
    return sizeof(size_t) + sizeof(T) * vec.size();
}

template <typename T>
inline void *serialization::serialize(void *out, const persistent_allocation<host, T>& in)
{
    size_t size = in.size();
    out = serialize(out, size);

    for(size_t i = 0; i < size; i++)
    {
        out = serialize(out, in[i]);
    }

    return out;
}

template <typename T>
inline void *serialization::unserialize(persistent_allocation<host, T> *out, void *in)
{
    size_t size;
    in = unserialize(&size, in);

    out->resize(size);

    for(size_t i = 0; i < size; i++)
    {
        in = unserialize(&((*out)[i]), in);
    }

    return in;
}


//// firepony packed vector
template <uint32 bits>
inline size_t serialization::serialized_size(const packed_vector<host, bits>& vec)
{
    return serialized_size(vec.m_size) + serialized_size(vec.m_storage);
}

template <uint32 bits>
inline void *serialization::serialize(void *out, const packed_vector<host, bits>& in)
{
    out = serialize(out, in.m_size);
    out = serialize(out, in.m_storage);

    return out;
}

template <uint32 bits>
inline void *serialization::unserialize(packed_vector<host, bits> *out, void *in)
{
    in = unserialize(&out->m_size, in);
    in = unserialize(&out->m_storage, in);

    return in;
}


//// string database
template <>
inline size_t serialization::serialized_size(const string_database& db)
{
    return serialized_size(db.string_identifiers);
}

template <>
inline void *serialization::serialize(void *out, const string_database& db)
{
    return serialize(out, db.string_identifiers);
}

template <>
inline void *serialization::unserialize(string_database *out, void *in)
{
    in = unserialize(&out->string_identifiers, in);

    out->string_hash_to_id.clear();

    for(uint32 i = 0; i < out->string_identifiers.size(); i++)
    {
        const std::string& str = out->string_identifiers[i];
        const uint32 h = string_database::hash(str);

        out->string_hash_to_id[h] = i;
    }

    return in;
}


//// sequence storage
template <>
inline size_t serialization::serialized_size(const sequence_storage<host>& data)
{
    return serialized_size(data.bases);
}

template <>
inline void *serialization::serialize(void *out, const sequence_storage<host>& data)
{
    return serialize(out, data.bases);
}

template <>
inline void *serialization::unserialize(sequence_storage<host> *data, void *in)
{
    in = unserialize(&data->bases, in);
    return in;
}


//// host sequence database
template <>
inline size_t serialization::serialized_size(const sequence_database_host& db)
{
    size_t ret = 0;

    ret += serialized_size(db.sequence_names);

    size_t size = db.storage.size();
    ret += serialized_size(size);
    for(size_t i = 0; i < size; i++)
    {
        ret += serialized_size(*(db.storage[i]));
    }

    return ret;
}

template <>
inline void *serialization::serialize(void *out, const sequence_database_host& db)
{
    out = serialize(out, db.sequence_names);

    out = serialize(out, db.storage.size());
    for(size_t i = 0; i < db.storage.size(); i++)
    {
        out = serialize(out, *(db.storage[i]));
    }

    return out;
}

template <>
inline void *serialization::unserialize(sequence_database_host *db, void *in)
{
    in = unserialize(&db->sequence_names, in);

    size_t size;
    in = unserialize(&size, in);

    for(size_t i = 0; i < size; i++)
    {
        auto *seq = db->new_entry(i);
        in = unserialize(seq, in);
    }

    return in;
}


//// variant storage
template <>
inline size_t serialization::serialized_size(const variant_storage<host>& v)
{
    size_t ret = 0;

    ret += serialized_size(v.feature_start);
    ret += serialized_size(v.feature_stop);
    ret += serialized_size(v.max_end_point_left);

    return ret;
}

template <>
inline void *serialization::serialize(void *out, const variant_storage<host>& v)
{
    out = serialize(out, v.feature_start);
    out = serialize(out, v.feature_stop);
    out = serialize(out, v.max_end_point_left);

    return out;
}

template <>
inline void *serialization::unserialize(variant_storage<host> *v, void *in)
{
    in = unserialize(&v->feature_start, in);
    in = unserialize(&v->feature_stop, in);
    in = unserialize(&v->max_end_point_left, in);

    return in;
}


//// variant database
template <>
inline size_t serialization::serialized_size(const variant_database_host& db)
{
    size_t ret = 0;

    ret += serialized_size(db.storage.size());
    for(size_t i = 0; i < db.storage.size(); i++)
    {
        ret += serialized_size(*(db.storage[i]));
    }

    return ret;
}

template <>
inline void *serialization::serialize(void *out, const variant_database_host& db)
{
    out = serialize(out, db.storage.size());
    for(size_t i = 0; i < db.storage.size(); i++)
    {
        out = serialize(out, *(db.storage[i]));
    }

    return out;
}

template <>
inline void *serialization::unserialize(variant_database_host *db, void *in)
{
    size_t size;
    in = unserialize(&size, in);

    for(size_t i = 0; i < size; i++)
    {
        auto *seq = db->new_entry(i);
        in = unserialize(seq, in);
    }

    return in;
}

} // namespace firepony
