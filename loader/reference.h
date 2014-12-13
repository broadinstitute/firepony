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

#include "../types.h"
#include "../sequence_data.h"

#include <fstream>
#include <map>

namespace firepony {

struct reference_file_handle
{
    const std::string filename;
    std::ifstream file_handle;

    // maps a string hash to an index in the reference file
    std::map<uint32, size_t> reference_index;
    bool index_available;

    sequence_data_host sequence_data;
    uint32 data_mask;

    bool make_sequence_available(const std::string& sequence_name);

    static reference_file_handle *open(const std::string filename, uint32 data_mask);

private:
    reference_file_handle(const std::string filename, uint32 data_mask)
        : filename(filename), data_mask(data_mask)
    { }

    bool load_index(void);
    void load_whole_reference(void);
};

} // namespace firepony
