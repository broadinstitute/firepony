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

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>

#include "mmap.h"

namespace firepony {

shared_memory_file::shared_memory_file()
    : fd(-1),
      size(0),
      data(nullptr)
{
}

void shared_memory_file::unmap(void)
{
    if (data)
    {
        munmap(data, size);
        data = nullptr;
    }

    size = 0;

    if (fd != -1)
    {
        close(fd);
        fd = -1;
    }
}

// given a file name, either relative or absolute, compute the shmem handle name
// returns "" in case of error
static std::string compute_shmem_path(const char *fname)
{
    std::string ret;
    char *abs_path = NULL;

    // convert fname into a full path name
    abs_path = realpath(fname, NULL);
    if (abs_path == NULL)
    {
        return std::string("");
    }

    // replace all slashes with underscores
    for(unsigned int i = 0; i < strlen(abs_path); i++)
    {
        if (abs_path[i] == '/')
        {
            abs_path[i] = '_';
        }
    }

    ret = std::string("/bqsr_") + std::string(abs_path);
    free(abs_path);

    return ret;
}

bool shared_memory_file::open(shared_memory_file *out, const char *fname)
{
    std::string shmem_path;
    struct stat st;
    int ret;

    // compute the shmem handle path
    shmem_path = compute_shmem_path(fname);
    if (shmem_path.size() == 0)
    {
        return false;
    }

    // open it
    out->fd = shm_open(shmem_path.c_str(), O_RDONLY, 0);
    if (out->fd == -1)
    {
        return false;
    }

    // figure out how big the shared memory block is
    ret = fstat(out->fd, &st);
    if (ret == -1)
    {
        close(out->fd);
        return false;
    }

    out->size = st.st_size;

    // map it into our address space
    int map_flags = MAP_SHARED;

    out->data = mmap(NULL, st.st_size, PROT_READ, map_flags, out->fd, 0);
    if (out->data == MAP_FAILED)
    {
        close(out->fd);
        return false;
    }

    return true;
}

bool shared_memory_file::create(shared_memory_file *out, const char *fname, size_t size)
{
    std::string shmem_path;
    int ret;

    // compute the shmem handle path
    shmem_path = compute_shmem_path(fname);
    if (shmem_path.size() == 0)
    {
        return false;
    }

    // open it, truncating if the file exists
    out->fd = shm_open(shmem_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (out->fd == -1)
    {
        return false;
    }

    ret = ftruncate(out->fd, size);
    if (ret == -1)
    {
        close(out->fd);
        return false;
    }

    out->size = size;

    // map it into our address space
    int map_flags = MAP_SHARED;

    out->data = mmap(NULL, out->size, PROT_READ | PROT_WRITE, map_flags, out->fd, 0);
    if (out->data == MAP_FAILED)
    {
        perror("mmap");
        close(out->fd);
        return false;
    }

    return true;
}

} // namespace firepony
