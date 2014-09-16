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

#include "string_database.h"


// returns the id of string if it exists in the database, otherwise returns -1
uint32 string_database::lookup(const std::string& string) const
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
const std::string& string_database::lookup(uint32 id) const
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
uint32 string_database::insert(const std::string& string)
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

uint32 string_database::hash(const char* s)
{
    uint32 h = 0;
    while (*s)
        h = h * 101 + (uint32) *s++;
    return h;
}

uint32 string_database::hash(const std::string& s)
{
    return hash(s.c_str());
}

size_t string_database::serialized_size(void)
{
    return serialization::serialized_size(string_identifiers);
}

void *string_database::serialize(void *out)
{
    return serialization::encode(out, &string_identifiers);
}

void *string_database::unserialize(void *in)
{
    in = serialization::decode(&string_identifiers, in);

    string_hash_to_id.clear();

    for(uint32 i = 0; i < string_identifiers.size(); i++)
    {
        const std::string& str = string_identifiers[i];
        const uint32 h = hash(str);

        string_hash_to_id[h] = i;
    }

    return in;
}
