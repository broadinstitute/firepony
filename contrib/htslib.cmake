# build htslib
set(htslib_PREFIX ${CMAKE_BINARY_DIR}/contrib/htslib-prefix)
ExternalProject_Add(htslib
    PREFIX ${htslib_PREFIX}
    #GIT_REPOSITORY "https://github.com/samtools/htslib.git"
    GIT_REPOSITORY "git://wilkins/broadinstitute/htslib.git"
    GIT_TAG "4c3406f05d9911d0dbd7c360d90db6d78800f1b5"
    # get rid of asserts, since some of them cause unresolved symbols...
    PATCH_COMMAND patch < ${CMAKE_SOURCE_DIR}/contrib/htslib-disable-asserts.patch
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ""
    )

include_directories(${htslib_PREFIX}/src/htslib)
set(htslib_LIB ${htslib_PREFIX}/src/htslib/libhts.a)
