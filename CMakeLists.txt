cmake_minimum_required(VERSION 3.20) # for add_link_options and implicit target directories.
project("lhm" C CXX)
include(CheckIncludeFileCXX)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wno-cast-function-type)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# sanitizers
option(LHM_SANITIZE_THREAD "lhm: enable thread sanitizer" OFF)
option(LHM_SANITIZE_ADDRESS "lhm: enable address sanitizer" OFF)
option(LHM_SANITIZE_UNDEFINED "lhm: enable undefined sanitizer" OFF)

# utils
option(LHM_BUILD_COMMON "lhm: build common utils library" ${LHM_STANDALONE})

option(LHM_WASM_MEM64 "lhm: use 64-bit memory in WASM builds" ON)

if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if(LHM_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LHM_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(LHM_TOOLS_INSTALL_DEFAULT ${LHM_STANDALONE})
endif()

#
# option list
#

# debug
option(LHM_ALL_WARNINGS "lhm: enable all compiler warnings" ON)
option(LHM_ALL_WARNINGS_3RD_PARTY "lhm: enable all compiler warnings in 3rd party libs" OFF)

# sanitizers
option(LHM_SANITIZE_THREAD "lhm: enable thread sanitizer" OFF)
option(LHM_SANITIZE_ADDRESS "lhm: enable address sanitizer" OFF)
option(LHM_SANITIZE_UNDEFINED "lhm: enable undefined sanitizer" OFF)

# utils
option(LHM_BUILD_COMMON "lhm: build common utils library" ${LHM_STANDALONE})

# extra artifacts
option(LHM_BUILD_TESTS "lhm: build tests" ${LHM_STANDALONE})
option(LHM_BUILD_EXAMPLES "lhm: build examples" ${LHM_STANDALONE})
option(LHM_BUILD_APP "lhm: build the unified binary" ${LHM_STANDALONE})

option(LHM_TESTS_INSTALL "lhm: install tests" OFF)

if(NOT DEFINED GGML_LHMFILE)
    set(GGML_LHMFILE_DEFAULT ON)
endif()

if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fmt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llama.cpp/ggml/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rapidjson/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/subprocess.h)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(HTTPLIB_COMPILE ON CACHE BOOL "")
add_subdirectory(thirdparty/cpp-httplib)

add_subdirectory(thirdparty/fmt)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt instead bundled")
set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "")
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "")
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "")
add_subdirectory(thirdparty/spdlog)

option(GFLAGS_BUILD_TESTING OFF)
add_subdirectory(thirdparty/gflags)

option(RAPIDJSON_BUILD_TESTS OFF)
option(RAPIDJSON_BUILD_EXAMPLES OFF)
add_subdirectory(thirdparty/rapidjson)

add_subdirectory(thirdparty/llama.cpp/ggml)

# Project include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lhm)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/json_compat)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/json)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/models)

# add_compile_options(-Werror)

add_subdirectory(common)
add_subdirectory(models)
add_subdirectory(src)

#
# install
#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LHM_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LHM_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Location of library files")
set(LHM_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Location of binary  files")
