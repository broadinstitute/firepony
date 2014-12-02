if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(BOOST_BUILD_TYPE "debug")
else()
  set(BOOST_BUILD_TYPE "release")
endif()

set(Boost_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib/boost-install)
set(Boost_PREFIX ${CMAKE_BINARY_DIR}/contrib/boost-prefix)

ExternalProject_Add(boost
    PREFIX ${Boost_PREFIX}
    GIT_REPOSITORY "https://github.com/boostorg/boost.git"
    GIT_TAG "boost-1.56.0"
#    URL "http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.bz2"
#    URL_HASH MD5=d6eef4b4cacb2183f2bf265a5a03a354
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ./bootstrap.sh --prefix=${Boost_INSTALL_PREFIX} --libdir=${Boost_INSTALL_PREFIX}
    BUILD_COMMAND ./b2 -j12
        --prefix=${Boost_INSTALL_PREFIX}
        --libdir=${Boost_INSTALL_PREFIX}/lib
        --threading=single,multi
        --link=static
        --variant=${BOOST_BUILD_TYPE}
        --without-python
        --without-container
        --without-context
        --without-coroutine
        --without-filesystem
        --without-graph
        --without-graph_parallel
        --without-iostreams
        --without-locale
        --without-log
        --without-math
        --without-mpi
        --without-python
        --without-regex
        --without-serialization
        --without-signals
        --without-system
        --without-thread
        --without-timer
        --without-wave

    INSTALL_COMMAND ./b2 install
        --prefix=${Boost_INSTALL_PREFIX}
        --libdir=${Boost_INSTALL_PREFIX}/lib
        --threading=single,multi
        --link=static
        --variant=${BOOST_BUILD_TYPE}
        --without-python
        --without-container
        --without-context
        --without-coroutine
        --without-filesystem
        --without-graph
        --without-graph_parallel
        --without-iostreams
        --without-locale
        --without-log
        --without-math
        --without-mpi
        --without-python
        --without-regex
        --without-serialization
        --without-signals
        --without-system
        --without-thread
        --without-timer
        --without-wave
        COMMAND ${PROJECT_SOURCE_DIR}/contrib/fix-boost-install.sh ${Boost_PREFIX} ${Boost_INSTALL_PREFIX}
    INSTALL_DIR ${Boost_INSTALL_PREFIX}
)

# update the boost prefix with the actual root of the source tree
set(Boost_PREFIX ${Boost_PREFIX}/src/boost)

set(Boost_LIBRARY_DIR ${Boost_INSTALL_PREFIX}/lib/boost)
set(Boost_INCLUDE_DIR ${Boost_INSTALL_PREFIX}/include)
set(Boost_BINARY_DIR ${Boost_INSTALL_PREFIX}/boost-install)
set(Boost_B2 ${Boost_PREFIX}/b2)
