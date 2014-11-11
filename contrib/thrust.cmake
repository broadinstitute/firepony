set(thrust_PREFIX ${CMAKE_BINARY_DIR}/contrib/thrust-prefix)

ExternalProject_Add(thrust
    PREFIX ${thrust_PREFIX}
    GIT_REPOSITORY "https://github.com/thrust/thrust.git"
    GIT_TAG "1.8.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

set(thrust_INCLUDE ${thrust_PREFIX}/src/thrust)
include_directories(${thrust_INCLUDE})
