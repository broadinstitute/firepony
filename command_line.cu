/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include "runtime_options.h"

namespace firepony {

struct runtime_options command_line_options;

static void usage(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: firepony <options> <input-file-name>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -r, --reference <genome-file-name>    Use <genome-file-name> as reference (required, fasta format)\n");
    fprintf(stderr, "  -s, --snp-database <dbsnp-file-name>  Use <dbsnp-file-name> as a SNP database\n");
    fprintf(stderr, "  --no-reference-mmap\n");
    fprintf(stderr, "  --no-snp-database-mmap                Do not attempt to use system shared memory for reference or dbSNP\n");
    fprintf(stderr, "  -d, --debug                           Enable debugging (*extremely* verbose)\n");
    fprintf(stderr, "  --disable-rounding                    Disable rounding on the output tables\n");
    fprintf(stderr, "  -b, --batch-size <n>                  Process input in batches of <n> reads\n");
    fprintf(stderr, "  --mmap                                Load reference/dbsnp from system shared memory if present\n");
    fprintf(stderr, "  -v, --verbose                         Verbose logging\n");
    fprintf(stderr, "  -o, --output <output-file-name>       File to write tabulated output to (default is stdout)\n");
    fprintf(stderr, "  --gpu-only                            Use only the CUDA GPU-accelerated backend\n");
    fprintf(stderr, "  --cpu-only                            Use only the CPU backend\n");
    fprintf(stderr, "  --cpu-threads                         Number of CPU worker threads to run\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "  http://github.com/broadinstitute/firepony\n");
    fprintf(stderr, "  subtil@gmail.com\n");
    fprintf(stderr, "\n");

    exit(1);
}

// check environment variables for default arguments if present
static void parse_env_vars(void)
{
    command_line_options.reference = getenv("FIREPONY_REFERENCE");
    if (command_line_options.reference)
    {
        command_line_options.reference = strdup(command_line_options.reference);
    }

    command_line_options.snp_database = getenv("FIREPONY_DBSNP");
    if (command_line_options.snp_database)
    {
        command_line_options.snp_database = strdup(command_line_options.snp_database);
    }

    char *backend = getenv("FIREPONY_BACKEND");
    if (backend)
    {
        if (!strcmp(backend, "cuda"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_cuda = true;
        }

        if (!strcmp(backend, "cpu"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_tbb = true;
        }
    }

    char *cpu_threads = getenv("FIREPONY_CPU_THREADS");
    if (cpu_threads)
    {
        errno = 0;
        command_line_options.cpu_threads = strtol(cpu_threads, NULL, 10);
        if (errno != 0)
        {
            fprintf(stderr, "WARNING: error parsing FIREPONY_CPU_THREADS\n");
            command_line_options.cpu_threads = -1;
        }
    }
}

void parse_command_line(int argc, char **argv)
{
    static const char *options_short = "r:s:db:o:v";
    static struct option options_long[] = {
            { "reference", required_argument, NULL, 'r' },
            { "snp-database", required_argument, NULL, 's' },
            { "no-reference-mmap", no_argument, NULL, 'k' },
            { "no-snp-database-mmap", no_argument, NULL, 'l' },
            { "debug", no_argument, NULL, 'd' },
            { "disable-rounding", no_argument, NULL, 'n' },
            { "batch-size", required_argument, NULL, 'b' },
            { "mmap", no_argument, NULL, 'm' },
            { "verbose", no_argument, NULL, 'v' },
            { "output", required_argument, NULL, 'o' },
            { "gpu-only", no_argument, NULL, 'g' },
            { "cpu-only", no_argument, NULL, 'c' },
            { "cpu-threads", required_argument, NULL, 't' },
            { 0 },
    };

    parse_env_vars();

    int ch;
    while((ch = getopt_long(argc, argv, options_short, options_long, NULL)) != -1)
    {
        switch(ch)
        {
        case 'r':
            // --reference, -r
            command_line_options.reference = strdup(optarg);
            break;

        case 's':
            // --snp-database, -s
            command_line_options.snp_database = strdup(optarg);
            break;

        case 'k':
            // --no-reference-mmap
            command_line_options.reference_use_mmap = false;
            break;

        case 'l':
            // --no-snp-database-mmap
            command_line_options.snp_database_use_mmap = false;
            break;

        case 'd':
            // -d, --debug
            command_line_options.debug = true;
            break;

        case 'n':
            // --disable-rounding
            command_line_options.disable_output_rounding = true;
            break;

        case 'b':
            // -b, --batch-size
            errno = 0;
            command_line_options.batch_size = strtol(optarg, NULL, 10);
            if (errno != 0)
            {
                fprintf(stderr, "error: invalid batch size\n");
                usage();
            }

            break;

        case 'm':
            // --mmap
            command_line_options.try_mmap = true;
            break;

        case 'v':
            // -v, --verbose
            command_line_options.verbose = true;
            break;

        case 'o':
            // -o, --output
            command_line_options.output = strdup(optarg);
            break;

        case 'g':
            // --gpu-only
            command_line_options.disable_all_backends();
            command_line_options.enable_cuda = true;
            break;

        case 'c':
            // --cpu-only
            command_line_options.disable_all_backends();
            command_line_options.enable_tbb = true;
            break;

        case 't':
            // --cpu-threads
            errno = 0;
            command_line_options.cpu_threads = strtol(optarg, NULL, 10);
            if (errno != 0)
            {
                fprintf(stderr, "error: invalid number of cpu threads\n");
                usage();
            }

            break;

        case '?':
        case ':':
        default:
            usage();
            break;
        }
    }

    // check that we got required arguments
    if (command_line_options.reference == nullptr)
    {
        fprintf(stderr, "error: missing reference file name\n\n");
        usage();
    }

    if (command_line_options.snp_database == nullptr)
    {
        fprintf(stderr, "error: missing SNP database file name\n\n");
        usage();
    }

    if (optind == argc)
    {
        fprintf(stderr, "error: missing input file name\n\n");
        usage();
    }

    if (optind != argc - 1)
    {
        fprintf(stderr, "error: extraneous arguments\n\n");
        usage();
    }

    command_line_options.input = argv[optind];
}

static void concat(std::string& out, const char *in)
{
    if (out.size())
    {
        out = out + std::string(" ");
    }

    out = out + std::string(in);
}

std::string canonical_command_line(void)
{
    std::string ret = "";
    char buf[1024];

    snprintf(buf, sizeof(buf), "-r %s", command_line_options.reference);
    concat(ret, buf);

    snprintf(buf, sizeof(buf), "-s %s", command_line_options.snp_database);
    concat(ret, buf);

    snprintf(buf, sizeof(buf), "-o %s", command_line_options.output ? command_line_options.output : "-");
    concat(ret, buf);

    if (command_line_options.batch_size != uint32(-1))
    {
        snprintf(buf, sizeof(buf), "--batch-size %d", command_line_options.batch_size);
        concat(ret, buf);
    }

    if (command_line_options.debug)
    {
        concat(ret, "--debug");
    }

    if (command_line_options.disable_output_rounding)
    {
        concat(ret, "--disable-rounding");
    }

    if (!command_line_options.enable_cuda)
    {
        concat(ret, "--cpu-only");
    }

    if (!command_line_options.enable_tbb)
    {
        concat(ret, "--gpu-only");
    }

    if (command_line_options.cpu_threads != -1)
    {
        snprintf(buf, sizeof(buf), "--cpu-threads %d", command_line_options.cpu_threads);
        concat(ret, buf);
    }

    if (command_line_options.try_mmap)
    {
        concat(ret, "--mmap");

        if (!command_line_options.reference_use_mmap)
        {
            concat(ret, "--no-reference-mmap");
        }

        if (!command_line_options.snp_database_use_mmap)
        {
            concat(ret, "--no-snp-database-mmap");
        }
    }

    if (command_line_options.verbose)
    {
        concat(ret, "--verbose");
    }

    return ret;
}

} // namespace firepony
