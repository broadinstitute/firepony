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

#include "bqsr_types.h"

#include <vector>
#include <string>
#include <map>

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

    // returns the id of string if it exists in the database, otherwise returns -1
    uint32 lookup(const std::string& string) const
    {
        uint32 hash = string_database::hash(string);
        auto id = string_hash_to_id.find(hash);

        if (id == string_hash_to_id.end())
        {
            return uint32(-1);
        } else {
            return id->second;
        }
    }

    // returns the string corresponding to the given integer id
    const std::string& lookup(uint32 id) const
    {
        static const std::string invalid("<invalid>");

        if (id < string_identifiers.size())
        {
            return string_identifiers[id];
        } else {
            return invalid;
        }
    };

    // inserts a string into the database, returning the new ID
    // if string already exists, returns the existing ID
    uint32 insert(const std::string& string)
    {
        uint32 hash = string_database::hash(string);
        uint32 id = lookup(string);

        if (id == uint32(-1))
        {
            // new string, assign an ID and store in the vector
            id = string_identifiers.size();

            string_hash_to_id[hash] = id;
            string_identifiers.push_back(std::string(string));
        }

        return id;
    };

    static uint32 hash(const char* s)
    {
        uint32 h = 0;
        while (*s)
            h = h * 101 + (uint32) *s++;
        return h;
    }

    static uint32 hash(const std::string& s)
    {
        return hash(s.c_str());
    }
};
