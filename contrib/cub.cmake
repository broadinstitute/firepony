set(cub_PREFIX ${CMAKE_BINARY_DIR}/contrib/cub-prefix)
ExternalProject_Add(cub
    PREFIX ${cub_PREFIX}
    GIT_REPOSITORY "https://github.com/nsubtil/cub.git"
    # 1.4.1 + warning fixes
    GIT_TAG "fix-signedness-warning"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD 0
    )

include_directories(${cub_PREFIX}/src/cub)
