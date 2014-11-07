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

#include <sys/types.h>
#include <vector>

#include "device_types.h"

namespace firepony {

struct shared_memory_file
{
    int fd;
    size_t size;
    void *data;

    shared_memory_file();
    void unmap(void);

    // open a shared memory segment and create a read-only mapping for it
    static bool open(shared_memory_file *out, const char *fname);
    // create a new shared memory segment of a given size with a read-write mapping
    static bool create(shared_memory_file *out, const char *fname, size_t size);
};

} // namespace firepony
