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

#include "device_types.h"
#include "covariate_table.h"

namespace firepony {

template <target_system system> void build_read_group_table(firepony_context<system>& context);
template <target_system system> void output_read_group_table(firepony_context<system>& context);

} // namespace firepony

