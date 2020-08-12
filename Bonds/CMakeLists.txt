add_library(bondsCpu bondsKernelsCpu.cpp)
#add_executable(bonds bondsEngine.cpp)
#target_link_libraries(bonds bondsCpu)
#set_target_properties(bonds
#  PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#)

if(BUILD_BENCHMARK)
    add_executable(benchmarkBonds benchmarkBonds.cpp)
    target_link_libraries(benchmarkBonds bondsCpu benchmark::benchmark)
    set_target_properties(benchmarkBonds
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkBonds PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endif()

if(BUILD_TEST)
    add_executable(testBonds testBonds.cpp)
    target_link_libraries(testBonds bondsCpu ${GTEST_BOTH_LIBRARIES})
    target_include_directories(testBonds SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
    )
    set_target_properties(testBonds
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_test(NAME testBonds COMMAND testBonds)
endif()

if(BUILD_HIP)
    add_library(bondsHip bondsKernelsGpu.cpp)
    if(HIP_COMPILER STREQUAL "nvcc")
        set_property(TARGET bondsHip PROPERTY CUDA_STANDARD 11)
        set_source_files_properties(bondsKernelsGpu.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
        target_link_libraries(bondsHip hip::device)
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(bondsHip --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    target_include_directories(bondsHip SYSTEM BEFORE
      PUBLIC
        ${ROCM_ROOT_PATH}/include
    )
    #target_link_libraries(bonds bondsCuda)
    #set_source_files_properties(bondsEngine.cpp PROPERTIES LANGUAGE CUDA)
    if(BUILD_BENCHMARK)
        target_link_libraries(benchmarkBonds bondsHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET benchmarkBonds PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(benchmarkBonds.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(benchmarkBonds hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(benchmarkBonds --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
    if(BUILD_TEST)
        target_link_libraries(testBonds bondsHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET testBonds PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(testBonds.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(testBonds hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(testBonds --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
endif()
