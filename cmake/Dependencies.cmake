# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT
find_package(Git REQUIRED)
if(NOT Git_FOUND)
    message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# Test dependencies
if(BUILD_TEST)
  # Google Test (https://github.com/google/googletest)
  message(STATUS "Downloading and building GTest.")
  set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
  download_project(
    PROJ           googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.8.1
    INSTALL_DIR    ${GTEST_ROOT}
    CMAKE_ARGS     -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    LOG_DOWNLOAD   TRUE
    LOG_CONFIGURE  TRUE
    LOG_BUILD      TRUE
    LOG_INSTALL    TRUE
    BUILD_PROJECT  TRUE
    ${UPDATE_DISCONNECTED_IF_AVAILABLE}
  )
  find_package(GTest REQUIRED)
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  # Google Benchmark (https://github.com/google/benchmark.git)
  message(STATUS "Downloading and building Google Benchmark.")
  if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
    # hip-clang cannot compile googlebenchmark for some reason
    set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
  endif()
  # Download, build and install googlebenchmark library
  set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark CACHE PATH "")
  download_project(
    PROJ           googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.4.0
    INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
    CMAKE_ARGS     -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
    LOG_DOWNLOAD   TRUE
    LOG_CONFIGURE  TRUE
    LOG_BUILD      TRUE
    LOG_INSTALL    TRUE
    BUILD_PROJECT  TRUE
    ${UPDATE_DISCONNECTED_IF_AVAILABLE}
  )
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()
