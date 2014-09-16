set(bqsr_sources
alignment_data.h
baq.cu
baq.h
bqsr_context.h
bqsr_context.cu
bqsr_types.h
cigar.h
cigar.cu
covariates.h
covariates_bit_packing.h
covariates_table.h
covariates_table.cu
covariates.cu
filters.h
filters.cu
gamgee_loader.cu
gamgee_loader.h
mmap.h
mmap.cu
sequence_data.h
sequence_data.cu
serialization.h
serialization_inl.h
string_database.h
string_database.cu
util.cu
util.h
variant_data.h
variant_data.cu
variants.cu
variants.h

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
add_dependencies(bqsr-loader gamgee zlib)
