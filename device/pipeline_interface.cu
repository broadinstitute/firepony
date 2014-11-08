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

    virtual std::string get_name(void) override;

    virtual void setup(const runtime_options *options,
                       alignment_header_host *h_header,
                       sequence_data_host *h_reference,
                       variant_database_host *h_dbsnp) override
    {
        size_t num_bytes;

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

        context = new firepony_context<system>(*options, *header, *reference, *dbsnp);
        batch = new alignment_batch<system>();
    }

    virtual void process_batch(const alignment_batch_host *h_batch) override
    {
        batch->download(h_batch);
        firepony_process_batch(*context, *batch);
    }

    virtual void finish(void) override
    {
        firepony_postprocess(*context);
    }

    virtual const pipeline_statistics& get_statistics(void) override
    {
        return context->stats;
    }

};
METHOD_INSTANTIATE(firepony_device_pipeline, setup);
METHOD_INSTANTIATE(firepony_device_pipeline, process_batch);
METHOD_INSTANTIATE(firepony_device_pipeline, finish);
METHOD_INSTANTIATE(firepony_device_pipeline, get_statistics);

#if ENABLE_CUDA_BACKEND
template<>
std::string firepony_device_pipeline<firepony::cuda>::get_name(void)
{
    cudaDeviceProp prop;
    int dev;
    int runtime_version;

    cudaRuntimeGetVersion(&runtime_version);
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "%s (%lu MB, CUDA %d.%d)\n",
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
template<>
std::string firepony_device_pipeline<firepony::tbb>::get_name(void)
{
    return std::string("CPU (Threading Building Blocks)");
}
#endif

firepony_pipeline *firepony_pipeline::create(target_system system)
{
    switch(system)
    {
#if ENABLE_CUDA_BACKEND
    case firepony::cuda:
        return new firepony_device_pipeline<firepony::cuda>();
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
    case firepony::tbb:
        return new firepony_device_pipeline<firepony::tbb>();
#endif

    default:
        return nullptr;
    }
}

} // namespace firepony
