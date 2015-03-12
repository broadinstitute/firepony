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

#include "table_formatter.h"
#include "device/util.h"

namespace firepony {

// adds a column to the table
void table_formatter::add_column(const std::string& name, output_format fmt, bool force_right_align)
{
    column_names.push_back(std::string(name));
    column_formats.push_back(fmt);
    column_widths.push_back(strlen(name.c_str()));

    if (force_right_align)
    {
        column_right_aligned.push_back(true);
    } else {
        switch(fmt)
        {
        case FMT_STRING:
        case FMT_CHAR:
            column_right_aligned.push_back(false);
            break;

        default:
            column_right_aligned.push_back(true);
            break;
        }
    }

    num_columns++;
}

// start processing a new row
void table_formatter::start_row(void)
{
    col_idx = 0;
    num_rows++;
}

// signals the end of the current row
void table_formatter::end_row(void)
{
    if (!preprocess)
    {
        printf("\n");
    }
}

// signals the end of the table
// during preprocessing, this initiates the output pass and triggers header output
void table_formatter::end_table(void)
{
    if (preprocess)
    {
        preprocess = false;

        // print the table header data
        printf("#:GATKTable:%d:%d:", num_columns, num_rows);
        for(uint32 i = 0; i < num_columns; i++)
        {
            switch(column_formats[i])
            {
            case FMT_STRING:
            case FMT_CHAR:
                printf("%%s:");
                break;

            case FMT_UINT64:
                printf("%%d:");
                break;

            case FMT_FLOAT_2:
                if (command_line_options.disable_output_rounding)
                {
                    printf("%%.64f");
                } else {
                    printf("%%.2f:");
                }

                break;

            case FMT_FLOAT_4:
                if (command_line_options.disable_output_rounding)
                {
                    printf("%%.64f");
                } else {
                    printf("%%.4f:");
                }

                break;
            }
        }
        printf(";\n");

        printf("#:GATKTable:%s:%s\n", table_name.c_str(), description.c_str());

        for(uint32 i = 0; i < num_columns; i++)
        {
            printf("%s", column_names[i].c_str());
            if (i == num_columns - 1)
            {
                printf("\n");
            } else {
                printf("  ");
            }
        }
    } else {
        printf("\n");
    }
}

template <>
void table_formatter::data(std::string val)
{
    if (preprocess)
    {
        // track column widths prior to output
        column_widths[col_idx] = max(size_t(column_widths[col_idx]), strlen(val.c_str()));
    } else {
        // output the column with the correct width
        char fmt_string[256];

        snprintf(fmt_string, sizeof(fmt_string), "%%%ds", (column_right_aligned[col_idx] ? 1 : -1) * column_widths[col_idx]);
        printf(fmt_string, val.c_str());

        if (col_idx < num_columns - 1)
        {
            printf("  ");
        }
    }

    col_idx++;
}

// process a data element
template <typename T>
void table_formatter::data(T val)
{
    char data[256];
    const char *data_fmt_string;

    switch(column_formats[col_idx])
    {
    case FMT_CHAR:
        data_fmt_string = "%c";
        break;

    case FMT_UINT64:
        data_fmt_string = "%lu";
        break;

    case FMT_FLOAT_2:
        if (command_line_options.disable_output_rounding)
        {
            data_fmt_string = "%.64f";
        } else {
            data_fmt_string = "%.2f";
            val = round_n(val, 2);
        }

        break;

    case FMT_FLOAT_4:
        if (command_line_options.disable_output_rounding)
        {
            data_fmt_string = "%.64f";
        } else {
            data_fmt_string = "%.4f";
            val = round_n(val, 4);
        }

        break;

    default:
        assert(!"can't happen");
        return;
    }

    snprintf(data, sizeof(data), data_fmt_string, val);

    this->data(std::string(data));
}

template void table_formatter::data<uint64>(uint64 val);
template void table_formatter::data<float>(float val);
template void table_formatter::data<double>(double val);
template void table_formatter::data<std::string>(std::string val);
template void table_formatter::data<char>(char val);

void table_formatter::data_int_as_string(int val)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%d", val);
    data(std::string(buf));
}

} // namespace firepony
