if (APPLE AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(GAMGEE_TOOLCHAIN_FLAGS "toolset=clang cxxflags=-stdlib=libc++ linkflags=libc++")
else()
    set(GAMGEE_TOOLCHAIN_FLAGS "toolset=gcc")
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(GAMGEE_BUILD_TYPE "debug")
else()
  set(GAMGEE_BUILD_TYPE "release")
endif()

set(gamgee_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-prefix)
ExternalProject_Add(gamgee
    PREFIX ${gamgee_PREFIX}
    DEPENDS boost htslib
    GIT_REPOSITORY "https://github.com/broadinstitute/gamgee.git"
    GIT_TAG "37213c0dd9c43476a0e29b0e688c6cb9d7738a8d"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${Boost_B2} -s BOOST_ROOT=${Boost_INSTALL_PREFIX} -s BOOST_BUILD_PATH=${Boost_PREFIX} ${GAMGEE_TOOLCHAIN_FLAGS} variant=${GAMGEE_BUILD_TYPE}
    INSTALL_COMMAND ""
    )

include_directories(${gamgee_PREFIX}/src/gamgee ${gamgee_PREFIX}/src/gamgee/lib/htslib)
# xxxnsubtil: compiler version is hardcoded, need to fix
set(gamgee_LIB_PATH "${gamgee_PREFIX}/src/gamgee/bin/gcc-4.9/${GAMGEE_BUILD_TYPE}/link-static/")
set(gamgee_LIB ${gamgee_LIB_PATH}/libgamgee.a ${htslib_LIB})
