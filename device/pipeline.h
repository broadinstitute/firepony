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

#include "firepony_context.h"

namespace firepony {

// the abstracted interface with non-device code
struct firepony_pipeline
{
    // returns a string with the name of the current pipeline
    virtual std::string get_name(void) = 0;

    virtual void setup(const runtime_options *options,
                       alignment_header_host *header,
                       sequence_data_host *h_reference,
                       variant_database_host *h_dbsnp) = 0;
    virtual void process_batch(const alignment_batch_host *batch) = 0;
    virtual void finish(void) = 0;

    virtual pipeline_statistics& statistics(void) = 0;

    static firepony_pipeline *create(target_system system);
};

} // namespace firepony
