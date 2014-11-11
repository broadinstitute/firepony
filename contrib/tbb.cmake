# build tbb
set(tbb_PREFIX ${CMAKE_BINARY_DIR}/contrib/tbb-prefix)
set(tbb_INSTALL ${CMAKE_BINARY_DIR}/contrib/tbb-install)

set(tbb_SRC ${tbb_PREFIX}/src/tbb)

ExternalProject_Add(tbb
    PREFIX ${tbb_PREFIX}
    URL "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb43_20140724oss_src.tgz"
    URL_MD5 0791e5fc7d11b27360080ea4521e32bb
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ${PROJECT_SOURCE_DIR}/contrib/tbb-install.sh ${tbb_SRC} ${tbb_INSTALL}
    LOG_INSTALL 1
    )

include_directories(${tbb_INSTALL}/include)
set(tbb_LIB ${tbb_INSTALL}/lib/libtbb.so)
