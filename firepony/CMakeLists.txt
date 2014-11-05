set(firepony_sources
alignment_data.h
baq.cu
baq.h
bqsr_context.cu
bqsr_context.h
bqsr_types.h
cigar.cu
cigar.h
covariates.h
covariates_table.cu
covariates_table.h
covariates.cu
fractional_errors.cu
fractional_errors.h
gamgee_loader.cu
gamgee_loader.h
mmap.cu
mmap.h
options.cu
options.h
read_filters.cu
read_filters.h
read_group_table.cu
read_group_table.h
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

covariates/bit_packers/bit_packing.h
covariates/bit_packers/context.h
covariates/bit_packers/cycle_illumina.h
covariates/bit_packers/event_tracker.h
covariates/bit_packers/quality_score.h
covariates/bit_packers/read_group.h
covariates/table_context.h
covariates/table_cycle_illumina.h
covariates/table_quality_scores.h

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

cuda_add_library(firepony-common ${firepony_sources})
add_dependencies(firepony-common gamgee zlib)

cuda_add_executable(firepony firepony.cu)
target_link_libraries(firepony firepony-common ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(firepony gamgee zlib)

cuda_add_executable(firepony-loader firepony-loader.cu)
target_link_libraries(firepony-loader firepony-common ${gamgee_LIB} ${zlib_LIB} ${SYSTEM_LINK_LIBRARIES})
add_dependencies(firepony-loader gamgee zlib cub)

