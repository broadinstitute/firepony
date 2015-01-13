/*
 * Firepony
 * Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "string_database.h"

namespace firepony {

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

size_t string_database::size(void) const
{
    return string_identifiers.size();
}

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

} // namespace firepony
