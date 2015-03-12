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

#include "types.h"
#include "command_line.h"

#include <string>
#include <vector>

namespace firepony {

struct table_formatter
{
    // data types for the columns
    typedef enum {
        // plain string
        FMT_STRING,
        // single character string
        FMT_CHAR,
        // integer
        FMT_UINT64,
        // floating point, 2 fractional digits
        FMT_FLOAT_2,
        // floating point, 4 fractional digits
        FMT_FLOAT_4,
        // floating point, no rounding (for debugging only)
    } output_format;

    std::string table_name;
    std::string description;

    uint32 num_columns;
    uint32 num_rows;

    std::vector<std::string> column_names;
    std::vector<output_format> column_formats;
    std::vector<uint32> column_widths;
    std::vector<bool> column_right_aligned;

    // describes the current state of the formatter object
    // output happens in two passes: during the first pass,
    // we gather size information for each column, then use that
    // during the second pass to properly pad each column while outputting
    uint32 col_idx;     // current column index
    bool preprocess;    // determines which output pass we're in

    table_formatter(const std::string& table_name, const std::string& description)
        : table_name(table_name), description(description), num_columns(0), num_rows(0), col_idx(0), preprocess(true)
    { }

    table_formatter(const std::string& table_name)
        : table_name(table_name), description(""), num_columns(0), num_rows(0), col_idx(0), preprocess(true)
    { }

    // adds a column to the table
    void add_column(const std::string& name, output_format fmt, bool force_right_align = false);

    // start processing a new row
    void start_row(void);
    // signals the end of the current row
    void end_row(void);

    // signals the end of the table
    // during preprocessing, this initiates the output pass and triggers header output
    void end_table(void);

    // process a data element
    template <typename T>
    void data(T val);

    // process an integer data element, converting to string
    void data_int_as_string(int val);
};

} // namespace firepony
