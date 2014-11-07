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

#include <thread>

#include "alignment_data.h"
#include "gamgee_loader.h"

namespace firepony {

struct io_thread
{
    static constexpr bool DISABLE_THREADING = false;
    static constexpr int NUM_BUFFERS = 3;

    alignment_batch_host batches[NUM_BUFFERS];
    volatile int put, get;
    volatile bool eof;

    gamgee_alignment_file file;
    uint32 data_mask;

    std::thread thread;

    io_thread(const char *fname, uint32 data_mask);

    void start(void);
    void join(void);

    alignment_batch_host& next_buffer(void);
    bool done(void);

private:
    int wrap(int val);
    void run(void);
};

} // namespace firepony
