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

#include "pipeline.h"

#include "alignment_data_device.h"
#include "sequence_data_device.h"
#include "variant_data_device.h"

#include "device/primitives/backends.h"

#include <thread>

#if ENABLE_TBB_BACKEND
#include <tbb/task_scheduler_init.h>
#endif

namespace firepony {

template <target_system system> void firepony_process_batch(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void firepony_postprocess(firepony_context<system>& context);

template <target_system system>
struct firepony_device_pipeline : public firepony_pipeline
{
    alignment_header<system> *header;
    sequence_data<system> *reference;
    variant_database<system> *dbsnp;

    firepony_context<system> *context;
    alignment_batch<system> *batch;

    io_thread *reader;

    std::thread thread;
    uint32 compute_device;

    firepony_device_pipeline(uint32 compute_device)
        : compute_device(compute_device)
    { }

    virtual std::string get_name(void) override;

    virtual target_system get_system(void) override
    {
        return system;
    }

    virtual int get_compute_device(void) override
    {
        return compute_device;
    }

    virtual pipeline_statistics& statistics(void) override
    {
        return context->stats;
    }

    virtual void setup(io_thread *reader,
                       const runtime_options *options,
                       alignment_header_host *h_header,
                       sequence_data_host *h_reference,
                       variant_database_host *h_dbsnp) override
    {
        size_t num_bytes;
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }

        this->reader = reader;

        header = new alignment_header<system>(*h_header);
        reference = new sequence_data<system>(*h_reference);
        dbsnp = new variant_database<system>(*h_dbsnp);

        header->download();

        num_bytes = reference->download();
        if (system == firepony::cuda)
        {
            fprintf(stderr, "downloaded %lu MB of reference data\n", num_bytes / (1024 * 1024));
        }

        num_bytes = dbsnp->download();
        if (system == firepony::cuda)
        {
            fprintf(stderr, "downloaded %lu MB of variant data\n", num_bytes / (1024 * 1024));
        }

        context = new firepony_context<system>(compute_device, *options, *header, *reference, *dbsnp);
        batch = new alignment_batch<system>();
    }

    virtual void start(void) override
    {
        thread = std::thread(&firepony_device_pipeline<system>::run, this);
    }

    virtual void join(void) override
    {
        thread.join();
    }

    {
    }

    virtual void postprocess(void) override
    {
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }

        firepony_postprocess(*context);
    }

private:
    void run(void)
    {
        if (system == cuda)
        {
            cudaSetDevice(compute_device);
        }

        timer<host> io_timer;
        alignment_batch_host *h_batch;

        for(;;)
        {
            // try to get a batch to work on
            io_timer.start();
            h_batch = reader->get_batch();
            io_timer.stop();
            statistics().io.add(io_timer);

            if (h_batch == nullptr)
            {
                // no more data, we're done
                break;
            }

            // download to the device
            batch->download(h_batch);

            // process the batch
            firepony_process_batch(*context, *batch);

            // return it to the reader for reuse
            reader->retire_batch(h_batch);

            if (!context->options.debug)
            {
                fprintf(stderr, ".");
                fflush(stderr);
            }
        }
    }
};

#if ENABLE_CUDA_BACKEND
template<>
std::string firepony_device_pipeline<firepony::cuda>::get_name(void)
{
    cudaDeviceProp prop;
    int runtime_version;

    cudaRuntimeGetVersion(&runtime_version);
    cudaSetDevice(compute_device);
    cudaGetDeviceProperties(&prop, compute_device);

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "%s (%lu MB, CUDA %d.%d)",
             prop.name, prop.totalGlobalMem / (1024 * 1024),
             runtime_version / 1000, runtime_version % 100);

    return std::string(buf);
}
#endif

#if ENABLE_CPP_BACKEND
template<>
std::string firepony_device_pipeline<firepony::cpp>::get_name(void)
{
    return std::string("CPU (C++ threads)");
}
#endif

#if ENABLE_OMP_BACKEND
template<>
std::string firepony_device_pipeline<firepony::omp>::get_name(void)
{
    return std::string("CPU (OpenMP)");
}
#endif

#if ENABLE_TBB_BACKEND
tbb::task_scheduler_init tbb_scheduler_init(tbb::task_scheduler_init::deferred);
static int num_tbb_threads = -1;

template<>
std::string firepony_device_pipeline<firepony::intel_tbb>::get_name(void)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "CPU (Intel TBB, %d threads)", num_tbb_threads);
    return std::string(buf);
}

#endif

firepony_pipeline *firepony_pipeline::create(target_system system, uint32 device)
{
    switch(system)
    {
#if ENABLE_CUDA_BACKEND
    case firepony::cuda:
        return new firepony_device_pipeline<firepony::cuda>(device);
#endif

#if ENABLE_CPP_BACKEND
    case firepony::cpp:
        return new firepony_device_pipeline<firepony::cpp>();
#endif

#if ENABLE_OMP_BACKEND
    case firepony::omp:
        return new firepony_device_pipeline<firepony::omp>();
#endif

#if ENABLE_TBB_BACKEND
    case firepony::intel_tbb:
        // reserve device threads for other devices and I/O
        num_tbb_threads = tbb::task_scheduler_init::default_num_threads() - device - 1;
        tbb_scheduler_init.initialize(num_tbb_threads);
        return new firepony_device_pipeline<firepony::intel_tbb>(num_tbb_threads);
#endif

    default:
        return nullptr;
    }
}

} // namespace firepony
