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

#pragma once

#include "types.h"

namespace firepony {

struct runtime_options
{
    // file names for reference, SNP database, input and output files
    const char *reference;
    const char *snp_database;
    const char *input;
    const char *output;

    // whether to attempt to load either the reference or SNP database via mmap
    bool reference_use_mmap;
    bool snp_database_use_mmap;

    // the batch size to use
    uint32 batch_size;

    // enable debugging
    bool debug;
    // disable rounding on the output tables
    bool disable_output_rounding;

    // enable/disable the various backends
    bool enable_cuda;
    bool enable_tbb;

    // number of CPU worker threads
    // (default is -1, meaning use all available cores)
    int cpu_threads;

    // enable the shared memory reference/dbsnp loader
    bool try_mmap;

    void disable_all_backends(void)
    {
        enable_cuda = false;
        enable_tbb = false;
    }

    runtime_options()
    {
        reference = nullptr;
        snp_database = nullptr;
        input = nullptr;
        output = nullptr;

        reference_use_mmap = true;
        snp_database_use_mmap = true;
        batch_size = uint32(-1);

        debug = false;
        disable_output_rounding = false;

        enable_cuda = true;
        enable_tbb = true;

        cpu_threads = -1;

        try_mmap = false;
    }
};

} // namespace firepony

