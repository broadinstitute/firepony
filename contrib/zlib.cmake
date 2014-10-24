# build zlib (used by bqsr)
set(zlib_PREFIX ${CMAKE_BINARY_DIR}/contrib/zlib-prefix)
ExternalProject_Add(zlib
    PREFIX ${zlib_PREFIX}
    GIT_REPOSITORY "https://github.com/madler/zlib.git"
    GIT_TAG "v1.2.8"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ${zlib_PREFIX}/src/zlib/configure --prefix=${zlib_PREFIX} --static
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ""
    )

include_directories(${zlib_PREFIX}/src)
set(zlib_LIB ${zlib_PREFIX}/src/zlib/libz.a)
