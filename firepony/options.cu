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

#include "options.h"

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
}

void parse_command_line(int argc, char **argv)
{
    static const char *options_short = "r:s:d";
    static struct option options_long[] = {
            { "reference", required_argument, NULL, 'r' },
            { "snp-database", required_argument, NULL, 's' },
            { "no-reference-mmap", no_argument, NULL, 'k' },
            { "no-snp-database-mmap", no_argument, NULL, 'l' },
            { "debug", no_argument, NULL, 'd' },
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
