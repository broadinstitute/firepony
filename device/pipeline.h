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

#include <lift/sys/compute_device.h>

#include "../types.h"
#include "../runtime_options.h"
#include "../alignment_data.h"
#include "../sequence_database.h"
#include "../variant_database.h"
#include "../io_thread.h"

#include "firepony_context.h"

namespace firepony {

// the abstracted interface with non-device code
struct firepony_pipeline
{
    // returns a string with the name of the current pipeline
    virtual std::string get_name(void) = 0;
    virtual size_t get_total_memory(void) = 0;
    virtual target_system get_system(void) = 0;
    virtual pipeline_statistics& statistics(void) = 0;

    virtual void setup(io_thread *reader,
                       const runtime_options *options,
                       alignment_header_host *header,
                       sequence_database_host *h_reference,
                       variant_database_host *h_dbsnp) = 0;

    virtual void start(void) = 0;
    virtual void join(void) = 0;

    virtual void gather_intermediates(firepony_pipeline *other) = 0;
    virtual void postprocess(void) = 0;

    // create a firepony pipeline object on the given compute device
    static firepony_pipeline *create(lift::compute_device *device);
};

} // namespace firepony
