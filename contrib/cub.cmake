set(cub_PREFIX ${CMAKE_BINARY_DIR}/contrib/cub-prefix)
ExternalProject_Add(cub
    PREFIX ${cub_PREFIX}
    GIT_REPOSITORY "https://github.com/nvlabs/cub.git"
    GIT_TAG "62d7334d15b27765d6d503f342c80aee9560bf8f"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD 1
    )

include_directories(${cub_PREFIX}/src/cub)
