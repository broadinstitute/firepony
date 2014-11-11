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

#include "../../types.h"

#include <sys/time.h>

namespace firepony {

template <target_system system>
struct timer
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

#if ENABLE_CUDA_BACKEND
template <>
struct timer<firepony::cuda>
{
    cudaEvent_t start_event, stop_event;

    timer()
    {
        cudaEventCreate(&start_event, cudaEventDefault);
        cudaEventCreate(&stop_event, cudaEventDefault);
    }

    ~timer()
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
