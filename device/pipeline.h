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

#include "../types.h"
#include "../runtime_options.h"
#include "../alignment_data.h"
#include "../sequence_data.h"
#include "../variant_data.h"
#include "../io_thread.h"

#include "firepony_context.h"

namespace firepony {

// the abstracted interface with non-device code
struct firepony_pipeline
{
    // returns a string with the name of the current pipeline
    virtual std::string get_name(void) = 0;
    virtual int get_compute_device(void) = 0;
    virtual target_system get_system(void) = 0;
    virtual pipeline_statistics& statistics(void) = 0;

    virtual void setup(io_thread *reader,
                       const runtime_options *options,
                       alignment_header_host *header,
                       sequence_data_host *h_reference,
                       variant_database_host *h_dbsnp) = 0;

    virtual void start(void) = 0;
    virtual void join(void) = 0;

    virtual void gather_intermediates(firepony_pipeline *other) = 0;
    virtual void postprocess(void) = 0;

    // the meaning of device depends on the target system:
    // for cuda, it identifies the device to use
    // for tbb, it contains the number of threads that we've reserved for other devices and IO
    static firepony_pipeline *create(target_system system, uint32 device);
};

} // namespace firepony
