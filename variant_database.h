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

#include "types.h"
#include "string_database.h"
#include "segmented_database.h"

#include <vector>

namespace firepony {

// data type for the segmented database
template <target_system system>
struct variant_storage
{
    persistent_allocation<system, uint32> feature_start;           // feature start position in the reference
    persistent_allocation<system, uint32> feature_stop;            // feature stop position in the reference

    // contains a prefix scan of the end points using max as the operator
    // each element encodes the maximum end point of any feature to the left of it
    persistent_allocation<system, uint32> max_end_point_left;

    template <target_system other_system>
    void copy(const variant_storage<other_system>& other)
    {
        feature_start.copy(other.feature_start);
        feature_stop.copy(other.feature_stop);
        // xxxnsubtil: recompute this instead of moving across PCIE?
        max_end_point_left.copy(other.max_end_point_left);
    }

    void free(void)
    {
        feature_start.free();
        feature_stop.free();
        max_end_point_left.free();
    }
};

// variant database storage reduces to a specialization of segmented_database_storage
template <target_system system>
using variant_database_storage = segmented_database_storage<system, variant_storage>;

typedef variant_database_storage<host> variant_database_host;

template <target_system system>
using variant_database_device = variant_database_storage<system>;

template <target_system system>
using variant_database = segmented_database<system, variant_database_host, variant_database_device>;

} // namespace firepony
