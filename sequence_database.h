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

typedef segmented_coordinate<uint32> sequence_coordinate;

// data type for the segmented database
template <target_system system>
struct sequence_storage : public segmented_storage<system>
{
    typedef segmented_storage<system> base;

    packed_vector<system, 4> bases;

    sequence_storage<system>& operator=(const sequence_storage<host>& other)
    {
        base::id = other.id;
        bases = other.bases;
        return *this;
    }

    struct const_view : public segmented_storage<system>::const_view
    {
        typename packed_vector<system, 4>::const_view bases;
    };

    operator const_view() const
    {
        const_view v;
        v.id = base::id;
        v.bases = bases;
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
        typename packed_vector<system, 4>::const_stream_type get_sequence_data(sequence_coordinate coord)
        {
            auto& d = base::const_view::get_chromosome(coord.id);
            return d.bases.offset(coord.coordinate);
        }
    };

    const_view view() const
    {
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

template <target_system system>
using sequence_database_device = sequence_database_storage<system>;

template <target_system system>
struct sequence_database
{
    const sequence_database_host& host;
    sequence_database_device<system> device;

    sequence_database(const sequence_database_host& host)
        : host(host)
    { }

    resident_segment_map empty_segment_map(void) const
    {
        return host.empty_segment_map();
    }

    void update_resident_set(const resident_segment_map& target_resident_set)
    {
        device.update_resident_set(host, target_resident_set);
    }
};

} // namespace firepony
