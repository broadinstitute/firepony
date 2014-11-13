set(thrust_PREFIX ${CMAKE_BINARY_DIR}/contrib/thrust-prefix)

ExternalProject_Add(thrust
    PREFIX ${thrust_PREFIX}
    GIT_REPOSITORY "https://github.com/thrust/thrust.git"
    # we're tracking branch 1.8.0
    GIT_TAG "c2863d38d107a225e9dae9bb4ff9a5d39b71ab3b"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

set(thrust_INCLUDE ${thrust_PREFIX}/src/thrust)
include_directories(${thrust_INCLUDE})
