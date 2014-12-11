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

#include <gamgee/sam_reader.h>
#include "../alignment_data.h"
#include "reference.h"

namespace firepony {

struct alignment_file
{
private:
    gamgee::SamReader<gamgee::SamIterator> file;
    gamgee::SamHeader gamgee_header;
    gamgee::SamIterator iterator;

    // map read group identifiers in tag data to read group names from the header
    // the read group name is either taken from the platform unit string if present, or else it's just the identifier itself
    std::map<std::string, std::string> read_group_id_to_name;

public:
    alignment_header_host header;

    alignment_file(const char *fname);
    ~alignment_file();

    bool next_batch(alignment_batch_host *batch, uint32 data_mask, reference_file_handle *reference, const uint32 batch_size = 100000);
    const char *get_sequence_name(uint32 id);
};

} // namespace firepony

