cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

#
# FinanceBench project
#
project(FinanceBench CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Build options
option(BUILD_CUDA "Build CUDA executables" ON)
option(BUILD_OPENCL "Build OpenCL executables" ON)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(BUILD_CUDA)
    enable_language("CUDA")
    add_definitions(-DBUILD_CUDA)
endif()

add_subdirectory(Black-Scholes)
add_subdirectory(Bonds)
add_subdirectory(Monte-Carlo)
add_subdirectory(Repo/CPU)

if(BUILD_CUDA)
    add_subdirectory(Repo/CUDA)
endif()

if(BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    add_subdirectory(Black-Scholes/OpenCL)
    add_subdirectory(Monte-Carlo/OpenCL)
endif()
