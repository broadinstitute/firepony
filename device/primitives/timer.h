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

    time_series& operator+=(const time_series& other)
    {
        elapsed_time += other.elapsed_time;
        return *this;
    }

    template <typename Timer>
    void add(const Timer& timer)
    {
        elapsed_time += timer.elapsed_time();
    }
};

} // namespace firepony
