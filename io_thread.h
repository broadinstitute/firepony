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

    gamgee_alignment_file file;
    uint32 data_mask;

    std::thread thread;

    io_thread(const char *fname, uint32 data_mask, const int consumers);
    ~io_thread();

    void start(void);
    void join(void);

    alignment_batch_host *get_batch(void);
    void retire_batch(alignment_batch_host *batch);

private:
    void run(void);
};

} // namespace firepony
