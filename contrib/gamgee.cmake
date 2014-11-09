set(gamgee_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-prefix)
set(gamgee_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-install)

if (USE_SYSTEM_BOOST)
    set(BOOST_DEPENDS "")
    set(BOOST_CMAKE_ARGS "")
else()
    set(BOOST_DEPENDS "boost")
    set(BOOST_CMAKE_ARGS
            -DBOOST_ROOT=${Boost_INSTALL_PREFIX}
            -DBoost_NO_SYSTEM_PATHS=ON)
endif()

ExternalProject_Add(gamgee
    PREFIX ${gamgee_PREFIX}
    DEPENDS ${BOOST_DEPENDS}
    GIT_REPOSITORY "https://github.com/broadinstitute/gamgee.git"
    GIT_TAG "a5187dc61f381f44740c7665e8ae4cb789c1130b"
    INSTALL_DIR ${gamgee_PREFIX}/install
    CMAKE_ARGS ${BOOST_CMAKE_ARGS}
               -DINSTALL_DEPENDENCIES=ON
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${gamgee_INSTALL_PREFIX}
    )

include_directories(${gamgee_INSTALL_PREFIX}/include)
set(gamgee_LIB ${gamgee_INSTALL_PREFIX}/lib/libgamgee.a ${gamgee_INSTALL_PREFIX}/lib/libhts.a)
