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

#include "io_thread.h"
#include "alignment_data.h"
#include "gamgee_loader.h"
#include "command_line.h"

namespace firepony {

io_thread::io_thread(const char *fname, uint32 data_mask)
    : put(1),
      get(0),
      eof(false),
      file(fname),
      data_mask(data_mask)
{ }

void io_thread::start(void)
{
    if (!DISABLE_THREADING)
    {
        thread = std::thread(&io_thread::run, this);
    }
}

void io_thread::join(void)
{
    if (!DISABLE_THREADING)
    {
        thread.join();
    }
}

int io_thread::wrap(int val)
{
    return val % NUM_BUFFERS;
}

alignment_batch_host& io_thread::next_buffer(void)
{
    if (DISABLE_THREADING)
    {
        return batches[0];
    } else {
        while(wrap(get + 1) == put)
            std::this_thread::yield();

        get = wrap(get + 1);
        return batches[get];
    }
}

bool io_thread::done(void)
{
    if (DISABLE_THREADING)
    {
        eof = !(file.next_batch(&batches[0], data_mask, command_line_options.batch_size));
        return eof;
    } else {
        if (!eof)
            return false;

        if (wrap(get + 1) != put)
            return false;

        return true;
    }
}

void io_thread::run(void)
{
    while(!eof)
    {
        // wait for a slot
        while (put == get)
            std::this_thread::yield();

        eof = !(file.next_batch(&batches[put], data_mask, command_line_options.batch_size));
        if (!eof)
        {
            put = wrap(put + 1);
        }
    }
}

} // namespace firepony
