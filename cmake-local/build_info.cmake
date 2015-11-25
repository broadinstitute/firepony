# sets our build information variables and generates build_info.cpp
site_name(BUILD_HOST)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git)
  if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE "BUILD_COMMIT"
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
  else (GIT_FOUND)
    set(BUILD_COMMIT 0)
  endif (GIT_FOUND)
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build_info.h.in" "${CMAKE_CURRENT_BINARY_DIR}/gen-include/build_info.h" @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/gen-include")
