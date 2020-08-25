# FinanceBench

HIP Implementation of FinanceBench project.

## Requirements

* Git
* CMake (3.5.1 or later)
* OpenMP
* For AMD GPUs:
  * AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
    * Including [HIP-clang](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang) compiler, which must be
      set as C++ compiler on ROCm platform.
* For NVIDIA GPUs:
  * CUDA Toolkit

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by CMake script.
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is enabled by default.
  * It will be automatically downloaded and built by cmake script.

## Build And Install

```shell
# Go to FinanceBench directory, create and go to the build directory.
cd FinanceBench; mkdir build; cd build

# Configure FinanceBench, setup options for your system.
# Build options:
#   BUILD_HIP - ON by default,
#   BUILD_TEST - ON by default,
#   BUILD_BENCHMARK - ON by default.
#
# ! IMPORTANT !
# Set C++ compiler to HCC or HIP-clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
[CXX=hipcc] cmake ../. # or cmake-gui ../.

# To configure FinanceBench for Nvidia platforms, 'CXX=<path-to-nvcc>', `CXX=nvcc` or omitting the flag
# entirely before 'cmake' is sufficient
[CXX=nvcc] cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or
cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or to build benchmarks
cmake -DBUILD_BENCHMARK=ON ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Package
make package

# Install
[sudo] make install
```

## Running Unit Tests

```shell
# Go to FinanceBench build directory
cd FinanceBench; cd build

# To run all tests
ctest

# To run unit tests for FinanceBench
./<unit-test-name>
```

## Running Benchmarks

```shell
# Go to FinanceBench build directory
cd FinanceBench; cd build

# To run benchmark:
# Further option can be found using --help
# [] Fields are optional
./benchmark<function_name> [--device_id <device_id>] [--size <size>] [--trials <trials>]
```
