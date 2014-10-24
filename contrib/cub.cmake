set(cub_PREFIX ${CMAKE_BINARY_DIR}/contrib/cub-prefix)
ExternalProject_Add(cub
    PREFIX ${cub_PREFIX}
    GIT_REPOSITORY "https://github.com/nvlabs/cub.git"
    GIT_TAG "70de43938d9bed686531a5bf01bd6cbba8959884"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

include_directories(${cub_PREFIX}/src/cub)
