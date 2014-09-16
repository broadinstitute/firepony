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
#include "alignment_data.h"

struct gamgee_file
{
private:
    gamgee::SamReader<gamgee::SamIterator> file;
    gamgee::SamHeader gamgee_header;
    gamgee::SamIterator iterator;

public:
    alignment_header header;

    gamgee_file(const char *fname);
    ~gamgee_file();

    bool next_batch(alignment_batch *batch, uint32 data_mask, const uint32 batch_size = 100000);
    const char *get_sequence_name(uint32 id);
};
