cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

#
# FinanceBench project
#
project(FinanceBench)

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

add_subdirectory(Black-Scholes/CPU)
add_subdirectory(Bonds/CPU)
add_subdirectory(Monte-Carlo/CPU)
add_subdirectory(Repo/CPU)

if(BUILD_CUDA)
    enable_language("CUDA")
    add_subdirectory(Black-Scholes/CUDA)
    add_subdirectory(Bonds/CUDA)
    add_subdirectory(Monte-Carlo/CUDA)
    add_subdirectory(Repo/CUDA)
endif()

if(BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    add_subdirectory(Black-Scholes/OpenCL)
    add_subdirectory(Monte-Carlo/OpenCL)
endif()
