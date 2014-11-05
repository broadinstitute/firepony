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

#include "types.h"

#include <sys/time.h>

namespace firepony {

// prepare temp_storage to store num_elements to be packed into a bit vector
void pack_prepare_storage_2bit(D_VectorU8& storage, uint32 num_elements);
void pack_prepare_storage_1bit(D_VectorU8& storage, uint32 num_elements);

// packs a vector of uint8 into a bit vector
void pack_to_2bit(D_PackedVector_2b& dest, D_VectorU8& src);
void pack_to_1bit(D_PackedVector_1b& dest, D_VectorU8& src);

// round a double to the Nth decimal place
double round_n(double val, int n);

// timers
struct cpu_timer
{
    struct timeval start_event, stop_event;

    void start(void)
    {
        gettimeofday(&start_event, NULL);
    }

    void stop(void)
    {
        gettimeofday(&stop_event, NULL);
    }

    float elapsed_time(void) const
    {
        struct timeval res;

        timersub(&stop_event, &start_event, &res);
        return res.tv_sec + res.tv_usec / 1000000.0;
    }
};

#ifdef RUN_ON_CPU
typedef struct cpu_timer device_timer;
#else
struct device_timer
{
    cudaEvent_t start_event, stop_event;

    device_timer()
    {
        cudaEventCreate(&start_event, cudaEventDefault);
        cudaEventCreate(&stop_event, cudaEventDefault);
    }

    ~device_timer()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(void)
    {
        cudaEventRecord(start_event);
    }

    void stop(void)
    {
        cudaEventRecord(stop_event);
    }

    float elapsed_time(void) const
    {
        float ms;

        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms / 1000.0;
    }

};
#endif

struct time_series
{
    float elapsed_time;

    time_series()
        : elapsed_time(0.0)
    { }

    template <typename Timer>
    void add(const Timer& timer)
    {
        elapsed_time += timer.elapsed_time();
    }
};

} // namespace firepony
