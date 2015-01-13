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

#include <thread>

#include "alignment_data.h"
#include "loader/alignments.h"
#include "loader/reference.h"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace firepony {

struct semaphore
{
    std::mutex sem_mutex;
    std::condition_variable cond_var;
    std::atomic<int> semaphore_value;

    semaphore()
        : semaphore_value(0)
    { }

    void wait(void)
    {
        // this is a little unfortunate, but we're forced to take the lock and immediately release it
        std::unique_lock<std::mutex> lock(sem_mutex);
        while (semaphore_value == 0)
        {
            cond_var.wait(lock);
        }

        semaphore_value--;
    }

    void post(void)
    {
        semaphore_value++;
        cond_var.notify_one();
    }
};

template <typename T>
struct locked_queue
{
    std::queue<T> queue;
    std::mutex m;

    size_t size(void)
    {
        std::lock_guard<std::mutex> lock(m);
        return queue.size();
    }

    T pop(void)
    {
        std::lock_guard<std::mutex> lock(m);

        T ret = queue.front();
        queue.pop();

        return ret;
    }

    void push(T elem)
    {
        std::lock_guard<std::mutex> lock(m);
        queue.push(elem);
    }
};

struct io_thread
{
    const int NUM_BUFFERS;

    semaphore sem_producer, sem_consumer;

    // a queue with batches ready for processing
    locked_queue<alignment_batch_host *> batches;
    // a queue containing processed batches for reuse
    locked_queue<alignment_batch_host *> empty_batches;

    // the reference file handle on which we'll load reference sequences
    reference_file_handle *reference;

    alignment_file file;
    uint32 data_mask;

    std::thread thread;

    io_thread(const char *fname, uint32 data_mask, const int consumers, reference_file_handle *reference);
    ~io_thread();

    void start(void);
    void join(void);

    alignment_batch_host *get_batch(void);
    void retire_batch(alignment_batch_host *batch);

private:
    void run(void);
};

} // namespace firepony
