set(cub_PREFIX ${CMAKE_BINARY_DIR}/contrib/cub-prefix)
ExternalProject_Add(cub
    PREFIX ${cub_PREFIX}
#    GIT_REPOSITORY "https://github.com/nvlabs/cub.git"
    GIT_REPOSITORY "git://wilkins/nvlabs/cub.git"
    GIT_TAG "a615c0f3b4a7dc78e9626e0a7c29d6e452da5230"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

include_directories(${cub_PREFIX}/src/cub)
