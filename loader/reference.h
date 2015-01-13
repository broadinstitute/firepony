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
#include "../sequence_data.h"

#include <fstream>
#include <map>
#include <mutex>
#include <deque>

namespace firepony {

struct reference_file_handle
{
    const std::string filename;
    std::ifstream file_handle;

    // maps a string hash to an index in the reference file
    std::map<uint32, size_t> reference_index;
    bool index_available;

    sequence_data_host sequence_data;
    uint32 data_mask;
    // list of mutexes that protect sequence_data, one per consumer thread
    std::deque<std::mutex> sequence_mutexes;
    // number of consumer threads
    const uint32 consumers;

    bool make_sequence_available(const std::string& sequence_name);

    static reference_file_handle *open(const std::string filename, uint32 data_mask, uint32 consumers);

    void consumer_lock(const uint32 consumer_id);
    void consumer_unlock(const uint32 consumer_id);

private:
    reference_file_handle(const std::string filename, uint32 data_mask, uint32 consumers);

    void producer_lock(void);
    void producer_unlock(void);

    bool load_index(void);
    void load_whole_reference(void);
};

} // namespace firepony
