set(gamgee_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-prefix)
set(gamgee_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-install)

ExternalProject_Add(gamgee
    PREFIX ${gamgee_PREFIX}
    DEPENDS boost
    GIT_REPOSITORY "https://github.com/broadinstitute/gamgee.git"
    GIT_TAG "3631525493724f7a05f82be389167dda27c9eedc"
    INSTALL_DIR ${gamgee_PREFIX}/install
    CMAKE_ARGS -DBOOST_ROOT=${Boost_INSTALL_PREFIX}
               -DBoost_NO_SYSTEM_PATHS=ON
               -DINSTALL_DEPENDENCIES=ON
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${gamgee_INSTALL_PREFIX}
    )

include_directories(${gamgee_INSTALL_PREFIX}/include)
set(gamgee_LIB ${gamgee_INSTALL_PREFIX}/lib/libgamgee.a ${gamgee_INSTALL_PREFIX}/lib/libhts.a)
