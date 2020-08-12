add_library(blackScholesCpu blackScholesAnalyticEngineKernelsCpu.cpp)
#add_executable(blackScholes blackScholesAnalyticEngine.cpp)
#target_link_libraries(blackScholes blackScholesCpu)
#set_target_properties(blackScholes
#  PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#)

if(BUILD_BENCHMARK)
    add_executable(benchmarkBlackScholes benchmarkBlackScholes.cpp)
    target_link_libraries(benchmarkBlackScholes blackScholesCpu benchmark::benchmark)
    set_target_properties(benchmarkBlackScholes
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkBlackScholes PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endif()

if(BUILD_TEST)
    add_executable(testBlackScholes testBlackScholes.cpp)
    target_link_libraries(testBlackScholes blackScholesCpu ${GTEST_BOTH_LIBRARIES})
    target_include_directories(testBlackScholes SYSTEM BEFORE
      PUBLIC
        ${GTEST_INCLUDE_DIRS}
    )
    set_target_properties(testBlackScholes
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_test(NAME testBlackScholes COMMAND testBlackScholes)
endif()

if(BUILD_HIP)
    add_library(blackScholesHip blackScholesAnalyticEngineKernelsGpu.cpp)
    if(HIP_COMPILER STREQUAL "nvcc")
        set_property(TARGET blackScholesHip PROPERTY CUDA_STANDARD 11)
        set_source_files_properties(blackScholesAnalyticEngineKernelsGpu.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
        target_link_libraries(blackScholesHip hip::device)
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(blackScholesHip --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    target_include_directories(blackScholesHip SYSTEM BEFORE
      PUBLIC
        ${ROCM_ROOT_PATH}/include
    )
    #target_link_libraries(blackScholes blackScholesHip)
    #set_source_files_properties(blackScholesAnalyticEngine.cpp PROPERTIES LANGUAGE CUDA)
    if(BUILD_BENCHMARK)
        target_link_libraries(benchmarkBlackScholes blackScholesHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET benchmarkBlackScholes PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(benchmarkBlackScholes.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(benchmarkBlackScholes hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(benchmarkBlackScholes --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
    if(BUILD_TEST)
        target_link_libraries(testBlackScholes blackScholesHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET testBlackScholes PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(testBlackScholes.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(testBlackScholes hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(testBlackScholes --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
endif()
