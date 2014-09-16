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

//#include <nvbio/basic/types.h>
//#include <nvbio/basic/vector.h>

#include "bqsr_types.h"

#include <vector>
#include <string>
#include <map>

uint32 bqsr_string_hash(const char* s);

// prepare temp_storage to store num_elements to be packed into a bit vector
void pack_prepare_storage_2bit(D_VectorU8& storage, uint32 num_elements);
void pack_prepare_storage_1bit(D_VectorU8& storage, uint32 num_elements);

// packs a vector of uint8 into a bit vector
void pack_to_2bit(D_PackedVector_2b& dest, D_VectorU8& src);
void pack_to_1bit(D_PackedVector_1b& dest, D_VectorU8& src);

// utility struct to keep track of string identifiers using integers
struct string_database
{
    std::vector<std::string> string_identifiers;
    std::map<uint32, uint32> string_hash_to_id;

    uint32 insert(const std::string& string)
    {
        uint32 hash = bqsr_string_hash(string.c_str());
        uint32 id;

        if (string_hash_to_id.find(hash) == string_hash_to_id.end())
        {
            // new string identifier, assign an ID and store in the vector
            id = string_identifiers.size();

            string_hash_to_id[hash] = id;
            string_identifiers.push_back(std::string(string));
        } else {
            // string already in the database, reuse the same ID
            id = string_hash_to_id[hash];
        }

        return id;
    };

    const std::string& lookup(uint32 id)
    {
        static const std::string invalid("<invalid>");

        if (id < string_identifiers.size())
        {
            return string_identifiers[id];
        } else {
            return invalid;
        }
    };
};

#include <thrust/scan.h>
#include <thrust/copy.h>

// simplified version of thrust::inclusive_scan
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline void bqsr_inclusive_scan(InputIterator first,
                                size_t len,
                                OutputIterator result,
                                Predicate op)
{
    thrust::inclusive_scan(first, first + len, result, op);
}

// simplified version of thrust::copy_if
template <typename InputIterator, typename OutputIterator, typename Predicate>
inline size_t bqsr_copy_if(InputIterator first,
                           size_t len,
                           OutputIterator result,
                           Predicate op)
{
    InputIterator last;
    last = thrust::copy_if(first, first + len, result, op);
    return last - result;
}
