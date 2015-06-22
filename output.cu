/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>

#include <ctime>

#include "output.h"

namespace firepony {

static FILE *output_fp = stdout;

bool output_open_file(const char *fname)
{
    output_fp = fopen(fname, "wt");
    if (output_fp == NULL)
    {
        fprintf(stderr, "error opening output file %s\n", fname);
        return false;
    }

    return true;
}

void output_printf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(output_fp, fmt, args);
    va_end(args);
}

static int last_progress_bar_len = -1;

void output_progress_bar(float progress, uint64_t batch_counter, std::time_t start)
{
    constexpr int LINE_WIDTH = 80;
    constexpr int BAR_SIZE = LINE_WIDTH - 40;
    int c;

    if (isatty(2))
    {
        if (last_progress_bar_len > 0)
        {
            for(c = 0; c < last_progress_bar_len; c++)
            {
                fprintf(stderr, "\b");
            }
        }
    }

    // draw the progress bar into a buffer
    char progress_bar[BAR_SIZE + 1];
    progress_bar[BAR_SIZE] = '\0';

    int bar_full_len = progress * (BAR_SIZE - 1);
    memset(progress_bar, '#', bar_full_len);
    if (bar_full_len < BAR_SIZE - 1)
        memset(&progress_bar[bar_full_len], ' ', BAR_SIZE - 1 - bar_full_len);

    char eta[1024];
    eta[0] = '\0';

    if (progress > 0.001)
    {
        // compute remaining time
        std::time_t now = std::time(nullptr);
        std::time_t elapsed = now - start;
        std::time_t total_time = elapsed / progress;
        std::time_t remaining = total_time - elapsed;

        char runtime[1024];
        strftime(runtime, sizeof(runtime), "%Hh %Mm", std::gmtime(&remaining));

        snprintf(eta, sizeof(eta), " ETA: %s", runtime);
    } else {
        eta[0] = '\0';
    }

    if (isatty(2))
    {
        last_progress_bar_len = fprintf(stderr, "[%s] %.02f%% %s", progress_bar, progress * 100.0, eta);
        fflush(stderr);
    } else {
        if (batch_counter % 50 == 0)
        {
            fprintf(stderr, "%.02f%% %s\n", progress * 100.0, eta);
        }
    }
}

} // namespace firepony
