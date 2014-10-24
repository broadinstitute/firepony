# build htslib
set(htslib_PREFIX ${CMAKE_BINARY_DIR}/contrib/htslib-prefix)
ExternalProject_Add(htslib
    PREFIX ${htslib_PREFIX}
    GIT_REPOSITORY "https://github.com/broadinstitute/htslib.git"
    # NOTE: gamgee tracks the 'broad' branch
    # we should always sync to a commit from that branch for consistency
    GIT_TAG "b1758c18004597ff970a5fc2e67953f3f2fc0df1"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ""
    )

include_directories(${htslib_PREFIX}/src/htslib)
set(htslib_LIB ${htslib_PREFIX}/src/htslib/libhts.a)
