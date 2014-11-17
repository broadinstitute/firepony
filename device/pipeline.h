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

    virtual void postprocess(void) = 0;

    // the meaning of device depends on the target system:
    // for cuda, it identifies the device to use
    // for tbb, it contains the number of threads that we've reserved for other devices and IO
    static firepony_pipeline *create(target_system system, uint32 device);
};

} // namespace firepony
