# FinanceBench

This is the improved implementation of the FinanceBench project, which was published in 2013. This code is part of an academic research project "*Accelerating financial applications on the GPU*", which was presented in the workshop "*GPGPU-6: Proceedings of the 6th Workshop on General Purpose Processor Using Graphics Processing*", March 2013. Citation of publication:
> Scott Grauer-Gray, William Killian, Robert Searles, and John Cavazos. 2013. Accelerating financial applications on the GPU. In Proceedings of the 6th Workshop on General Purpose Processor Using Graphics Processing Units (GPGPU-6). Association for Computing Machinery, New York, NY, USA, 127–136. DOI:https://doi.org/10.1145/2458523.2458536

The original code supported several backends. Currently only OpenMP for CPUs and CUDA/HIP for Nvidia and AMD GPUs are supported, while we improve performance.

## Requirements

To build te code, you need:

* Git
* CMake (3.5.1 or later)
* OpenMP
* For AMD GPUs:
  * AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
    * Including [HIP-clang](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang) compiler, which must be
      set as C++ compiler on ROCm platform.
* For NVIDIA GPUs:
  * CUDA Toolkit
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is enabled by default.
  * It will be automatically downloaded and built by cmake script.
* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by CMake script.

## Build And Install

Below is the explanation how to build with Linux and `make`. When you use Windows, make sure you use cmake-gui and build for your Visual Studio version. The variables to change are the same.

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
