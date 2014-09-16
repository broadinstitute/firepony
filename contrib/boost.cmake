if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(BOOST_BUILD_TYPE "debug")
else()
  set(BOOST_BUILD_TYPE "release")
endif()

set(Boost_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib/boost-install)
set(Boost_PREFIX ${CMAKE_BINARY_DIR}/contrib/boost-prefix)

ExternalProject_Add(boost
    PREFIX ${Boost_PREFIX}
    #GIT_REPOSITORY "https://github.com/boostorg/boost.git"
    GIT_REPOSITORY "git://wilkins/boostorg/boost.git"
    GIT_TAG "boost-1.56.0"
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ./bootstrap.sh
    BUILD_COMMAND ./b2 install -j12
        --prefix=${Boost_INSTALL_PREFIX}
        --threading=single,multi
        --link=static
        --variant=${BOOST_BUILD_TYPE}
        --without-python
    INSTALL_COMMAND ""
    INSTALL_DIR ${BOOST_INSTALL_PREFIX}
)

# update the boost prefix with the actual root of the source tree
set(Boost_PREFIX ${Boost_PREFIX}/src/boost)

if( NOT WIN32 )
    set(Boost_LIBRARY_DIR ${Boost_INSTALL_PREFIX}/lib/boost )
    set(Boost_INCLUDE_DIR ${Boost_INSTALL_PREFIX}/include )
    set(Boost_BINARY_DIR ${Boost_INSTALL_PREFIX}/boost-install )
    set(Boost_B2 ${Boost_PREFIX}/b2)
else()
    set(Boost_LIBRARY_DIR ${Boost_INSTALL_PREFIX}/lib/ )
    set(Boost_INCLUDE_DIR ${Boost_INSTALL_PREFIX}/include/boost-1_56/ )
    set(Boost_BINARY_DIR ${Boost_INSTALL_PREFIX}/boost-install )
    set(Boost_B2 ${Boost_PREFIX}/b2.exe)    
endif()
