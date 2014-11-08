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

namespace firepony {

struct runtime_options
{
    // file names for reference, SNP database and input files
    const char *reference;
    const char *snp_database;
    const char *input;

    // whether to attempt to load either the reference or SNP database via mmap
    bool reference_use_mmap;
    bool snp_database_use_mmap;

    // the batch size to use
    uint32 batch_size;

    // enable debugging
    bool debug;

    // enable/disable the various backends
    bool enable_cuda;
    bool enable_cpp;
    bool enable_omp;

    void disable_all_backends(void)
    {
        enable_cuda = false;
        enable_cpp = false;
        enable_omp = false;
    }

    runtime_options()
    {
        reference = nullptr;
        snp_database = nullptr;
        input = nullptr;

        reference_use_mmap = true;
        snp_database_use_mmap = true;
        batch_size = 20000;

        debug = false;

        enable_cuda = true;
        enable_cpp = true;
        enable_omp = true;
    }
};

} // namespace firepony

