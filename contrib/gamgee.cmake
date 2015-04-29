set(gamgee_PREFIX ${CMAKE_BINARY_DIR}/contrib/gamgee-prefix)
set(gamgee_INSTALL ${CMAKE_BINARY_DIR}/contrib/gamgee-install)

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
    GIT_TAG "d923b5301e477e2f42347a202482e6cd1428c635"
    UPDATE_COMMAND ""
    INSTALL_DIR ${gamgee_INSTALL}
    # relax the gamgee CMake version requirements
    PATCH_COMMAND patch -p1 -t -N < ${PROJECT_SOURCE_DIR}/contrib/gamgee-fix-cmake-version.patch
    CMAKE_ARGS ${BOOST_CMAKE_ARGS}
               -DINSTALL_DEPENDENCIES=ON
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${gamgee_INSTALL}
    LOG_DOWNLOAD 1
    LOG_INSTALL 1
    )

include_directories(${gamgee_INSTALL}/include)
set(gamgee_LIB ${gamgee_INSTALL}/lib/libgamgee.a ${gamgee_INSTALL}/lib/libhts.a)
