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

#pragma once

#include "types.h"
#include "string_database.h"
#include "segmented_database.h"

namespace firepony {

// sequence storage for a single chromosome
template <target_system system>
struct sequence_storage
{
    packed_vector<system, 4> bases;

    sequence_storage<system>& operator=(const sequence_storage<host>& other)
    {
        bases = other.bases;
        return *this;
    }

    struct const_view
    {
        typename packed_vector<system, 4>::const_view bases;
    };

    operator const_view() const
    {
        const_view v = {
            bases,
        };

        return v;
    }
};

template <target_system system>
struct sequence_database_storage : public segmented_database_storage<system, sequence_storage>
{
    // shorthand for base type
    typedef segmented_database_storage<system, sequence_storage> base;
    using base::storage;
    using base::views;

    struct const_view : public base::const_view
    {
        // grab a reference to the sequence stream at a given coordinate
        CUDA_HOST_DEVICE
        typename packed_vector<system, 4>::const_stream_type get_sequence_data(uint16 id, uint32 offset)
        {
            auto& d = base::const_view::get_chromosome(id);
            return d.bases.offset(offset);
        }
    };

    const_view view() const
    {
        base::update_views();

        const_view v;
        v.data = views;
        return v;
    }

    operator const_view() const
    {
        return view();
    }
};

struct sequence_database_host : public sequence_database_storage<host>
{
    // host-only data
    string_database sequence_names;
};

// device storage is identical to the generic version
template <target_system system>
using sequence_database_device = sequence_database_storage<system>;

// define the sequence_database type as a segmented_database for the storage types defined above
template <target_system system>
using sequence_database = segmented_database<system, sequence_database_host, sequence_database_device>;

} // namespace firepony
