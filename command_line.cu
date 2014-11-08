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
    fprintf(stderr, "usage: firepony <options> <input-file-name>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -r, --reference <genome-file-name>    Use <genome-file-name> as reference (required, fasta format)\n");
    fprintf(stderr, "  -s, --snp-database <dbsnp-file-name>  Use <dbsnp-file-name> as a SNP database\n");
    fprintf(stderr, "  --no-reference-mmap\n");
    fprintf(stderr, "  --no-snp-database-mmap                Do not attempt to use system shared memory for reference or dbSNP\n");
    fprintf(stderr, "  -d, --debug                           Enable debugging (*extremely* verbose)\n");
    fprintf(stderr, "  -b, --batch-size <n>                  Process input in batches of <n> reads\n");
#if ENABLE_CUDA_BACKEND
    fprintf(stderr, "  --gpu-only                            Use only the CUDA GPU-accelerated backend\n");
#endif
#if ENABLE_CPP_BACKEND
    fprintf(stderr, "  --cpp-only                            Use only the CPP threads CPU backend\n");
#endif
#if ENABLE_OMP_BACKEND
    fprintf(stderr, "  --openmp-only                         Use only the OpenMP CPU backend\n");
#endif
#if ENABLE_TBB_BACKEND
    fprintf(stderr, "  --tbb-only                            Use only the Threading Building Blocks CPU backend\n");
#endif
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
#if ENABLE_CUDA_BACKEND
        if (!strcmp(backend, "cuda"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_cuda = true;
        }
#endif

#if ENABLE_CPP_BACKEND
        if (!strcmp(backend, "cpp"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_cpp = true;
        }
#endif

#if ENABLE_OMP_BACKEND
        if (!strcmp(backend, "omp"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_omp = true;
        }
#endif

#if ENABLE_TBB_BACKEND
        if (!strcmp(backend, "tbb"))
        {
            command_line_options.disable_all_backends();
            command_line_options.enable_tbb = true;
        }
#endif
    }
}

void parse_command_line(int argc, char **argv)
{
    static const char *options_short = "r:s:db:";
    static struct option options_long[] = {
            { "reference", required_argument, NULL, 'r' },
            { "snp-database", required_argument, NULL, 's' },
            { "no-reference-mmap", no_argument, NULL, 'k' },
            { "no-snp-database-mmap", no_argument, NULL, 'l' },
            { "debug", no_argument, NULL, 'd' },
            { "batch-size", required_argument, NULL, 'b' },
#if ENABLE_CUDA_BACKEND
            { "gpu-only", no_argument, NULL, 'g' },
#endif
#if ENABLE_CPP_BACKEND
            { "cpp-only", no_argument, NULL, 'c' },
#endif
#if ENABLE_OMP_BACKEND
            { "omp-only", no_argument, NULL, 'm' },
#endif
#if ENABLE_OMP_BACKEND
            { "tbb-only", no_argument, NULL, 't' },
#endif
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

#if ENABLE_CUDA_BACKEND
        case 'g':
            // --gpu-only
            command_line_options.disable_all_backends();
            command_line_options.enable_cuda = true;
            break;
#endif

#if ENABLE_CPP_BACKEND
        case 'c':
            // --cpp-only
            command_line_options.disable_all_backends();
            command_line_options.enable_cpp = true;
            break;
#endif

#if ENABLE_OMP_BACKEND
        case 'm':
            // --omp-only
            command_line_options.disable_all_backends();
            command_line_options.enable_omp = true;
            break;
#endif

#if ENABLE_TBB_BACKEND
        case 't':
            // --tbb-only
            command_line_options.disable_all_backends();
            command_line_options.enable_tbb = true;
            break;
#endif

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

} // namespace firepony