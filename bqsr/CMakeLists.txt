set(bqsr_sources
alignment_data.h
baq.cu
baq.h
bqsr_context.cu
bqsr_context.h
bqsr_types.h
cigar.cu
cigar.h
covariates.h
covariates_bit_packing.h
covariates_table.cu
covariates_table.h
covariates.cu
fractional_errors.cu
fractional_errors.h
gamgee_loader.cu
gamgee_loader.h
mmap.cu
mmap.h
read_filters.cu
read_filters.h
sequence_data.cu
sequence_data.h
serialization.h
serialization_inl.h
snp_filter.cu
snp_filter.h
string_database.cu
string_database.h
util.cu
util.h
variant_data.cu
variant_data.h

from_nvbio/alphabet_inl.h
from_nvbio/alphabet.h
from_nvbio/bufferedtextfile.h
from_nvbio/dna.h

primitives/algorithms.h
primitives/cuda.h
primitives/packed_stream_packer.h
primitives/packed_stream.h
primitives/packed_vector.h
primitives/parallel.h
primitives/util.h
primitives/vector.h
)

cuda_add_library(bqsr-common ${bqsr_sources})
add_dependencies(bqsr-common gamgee zlib)

cuda_add_executable(bqsr bqsr.cu)
target_link_libraries(bqsr bqsr-common rt ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(bqsr gamgee zlib)

cuda_add_executable(bqsr-loader bqsr-loader.cu)
target_link_libraries(bqsr-loader bqsr-common rt ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(bqsr-loader gamgee zlib cub)
