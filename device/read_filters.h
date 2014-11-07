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

#include "firepony_context.h"

namespace firepony {

template <target_system system> void filter_reads(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void filter_bases(firepony_context<system>& context, const alignment_batch<system>& batch);

} // namespace firepony

