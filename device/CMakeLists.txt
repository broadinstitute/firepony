set(firepony_sources
alignment_data_device.h
baq.cu
baq.h
cigar.cu
cigar.h
covariate_table.cu
covariate_table.h
covariates.h
covariates.cu
empirical_quality.cu
empirical_quality.h
expected_error.h
firepony_context.cu
firepony_context.h
fractional_errors.cu
fractional_errors.h
pipeline.cu
pipeline_interface.cu
pipeline.h
read_filters.cu
read_filters.h
read_group_table.cu
read_group_table.h
snp_filter.cu
snp_filter.h
util.cu
util.h
variant_data_device.h

covariates/bit_packers/bit_packing.h
covariates/bit_packers/context.h
covariates/bit_packers/cycle_illumina.h
covariates/bit_packers/event_tracker.h
covariates/bit_packers/quality_score.h
covariates/bit_packers/read_group.h
covariates/packer_context.h
covariates/packer_cycle_illumina.h
covariates/packer_quality_score.h

from_nvbio/alphabet_inl.h
from_nvbio/alphabet.h
from_nvbio/dna.h

primitives/algorithms.h
primitives/backends.h
primitives/cuda.h
primitives/packed_stream_packer.h
primitives/packed_stream.h
primitives/packed_vector.h
primitives/parallel.h
primitives/util.h
primitives/vector.h
)

cuda_add_library(firepony-device ${firepony_sources})
add_dependencies(firepony-device gamgee zlib cub)
if (ENABLE_CPU_BACKEND)
    add_dependencies(firepony-device tbb)
endif()
