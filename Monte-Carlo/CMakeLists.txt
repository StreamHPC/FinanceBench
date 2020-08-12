add_library(monteCarloCpu monteCarloKernelsCpu.cpp)
#add_executable(monteCarlo monteCarloEngine.cpp)
#target_link_libraries(monteCarlo monteCarloCpu)
#set_target_properties(monteCarlo
#  PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#)

if(BUILD_BENCHMARK)
    add_executable(benchmarkMonteCarlo benchmarkMonteCarlo.cpp)
    target_link_libraries(benchmarkMonteCarlo monteCarloCpu benchmark::benchmark)
    set_target_properties(benchmarkMonteCarlo
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkMonteCarlo PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endif()

if(BUILD_TEST)
    add_executable(testMonteCarlo testMonteCarlo.cpp)
    target_link_libraries(testMonteCarlo monteCarloCpu ${GTEST_BOTH_LIBRARIES})
    target_include_directories(testMonteCarlo SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
    )
    set_target_properties(testMonteCarlo
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_test(NAME testMonteCarlo COMMAND testMonteCarlo)
endif()

if(BUILD_HIP)
    add_library(monteCarloHip monteCarloKernelsGpu.cpp)
    if(HIP_COMPILER STREQUAL "nvcc")
        set_property(TARGET monteCarloHip PROPERTY CUDA_STANDARD 11)
        set_source_files_properties(monteCarloKernelsGpu.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
        target_link_libraries(monteCarloHip hip::device)
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(monteCarloHip --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    target_include_directories(monteCarloHip SYSTEM BEFORE
      PUBLIC
        ${ROCM_ROOT_PATH}/include
        ${ROCM_ROOT_PATH}/hiprand/include
        ${ROCM_ROOT_PATH}/hiprand/include/hiprand
        ${ROCM_ROOT_PATH}/rocrand/include
    )
    #target_link_libraries(monteCarlo monteCarloCuda)
    #set_source_files_properties(monteCarloEngine.cpp PROPERTIES LANGUAGE CUDA)
    if(BUILD_BENCHMARK)
        target_link_libraries(benchmarkMonteCarlo monteCarloHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET benchmarkMonteCarlo PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(benchmarkMonteCarlo.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(benchmarkMonteCarlo hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(benchmarkMonteCarlo --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
    if(BUILD_TEST)
        target_link_libraries(testMonteCarlo monteCarloHip)
        if(HIP_COMPILER STREQUAL "nvcc")
            set_property(TARGET testMonteCarlo PROPERTY CUDA_STANDARD 11)
            set_source_files_properties(testMonteCarlo.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
            target_link_libraries(testMonteCarlo hip::device)
            foreach(amdgpu_target ${AMDGPU_TARGETS})
                target_link_libraries(testMonteCarlo --amdgpu-target=${amdgpu_target})
            endforeach()
        endif()
    endif()
endif()
