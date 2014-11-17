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

#include <mutex>
#include <condition_variable>

namespace firepony {

io_thread::io_thread(const char *fname, uint32 data_mask, const int consumers)
    : NUM_BUFFERS(consumers + 1),
      file(fname),
      data_mask(data_mask)
{
    for(int i = 0; i < NUM_BUFFERS; i++)
    {
        empty_batches.push(new alignment_batch_host);
    }
}

io_thread::~io_thread()
{
    while(empty_batches.size())
    {
        alignment_batch_host *buf = empty_batches.pop();
        if (buf)
            delete buf;
    }
}

void io_thread::start(void)
{
    thread = std::thread(&io_thread::run, this);
}

void io_thread::join(void)
{
    thread.join();
}

alignment_batch_host *io_thread::get_batch(void)
{
    // wait for a buffer to become available
    sem_consumer.wait();

    return batches.pop();
}

void io_thread::retire_batch(alignment_batch_host *buffer)
{
    empty_batches.push(buffer);

    // notify the producer
    sem_producer.post();
}

void io_thread::run(void)
{
    alignment_batch_host *buf;
    bool eof;

    // prime the buffer queue
    for(int i = 0; i < NUM_BUFFERS; i++)
    {
        assert(empty_batches.size());
        buf = empty_batches.pop();

        eof = !(file.next_batch(buf, data_mask, command_line_options.batch_size));
        if (eof)
        {
            break;
        }

        batches.push(buf);
        sem_consumer.post();
    }

    while(!eof)
    {
        // wait for a slot
        sem_producer.wait();

        assert(empty_batches.size());
        buf = empty_batches.pop();

        eof = !(file.next_batch(buf, data_mask, command_line_options.batch_size));
        if (!eof)
        {
            batches.push(buf);
            sem_consumer.post();
        }
    }

    // push null pointers into the queue to signal consumers we're done
    for(int i = 0; i < NUM_BUFFERS; i++)
    {
        batches.push(nullptr);
        sem_consumer.post();
    }
}

} // namespace firepony
