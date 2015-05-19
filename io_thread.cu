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

#include "io_thread.h"
#include "alignment_data.h"
#include "loader/alignments.h"
#include "command_line.h"

#include <mutex>
#include <condition_variable>

namespace firepony {

io_thread::io_thread(const char *fname, uint32 data_mask, const int consumers, reference_file_handle *reference)
    : NUM_BUFFERS(consumers + 1),
      file(fname),
      data_mask(data_mask),
      reference(reference)
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

bool io_thread::start(void)
{
    if (file.init() == false)
        return false;

    thread = std::thread(&io_thread::run, this);

    return true;
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

        eof = !(file.next_batch(buf, data_mask, reference, command_line_options.batch_size));
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

        eof = !(file.next_batch(buf, data_mask, reference, command_line_options.batch_size));
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
