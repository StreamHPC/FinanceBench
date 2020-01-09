add_library(bondsCpu bondsKernelsCpu.cpp)
add_executable(bonds bondsEngine.cpp)
target_link_libraries(bonds bondsCpu)
set_target_properties(bonds
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(bondsCuda bondsKernelsGpu.cu)
    target_link_libraries(bonds bondsCuda)
    set_source_files_properties(bondsEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(BUILD_BENCHMARK)
    add_executable(benchmarkBonds benchmarkBonds.cpp)
    target_link_libraries(benchmarkBonds bondsCpu bondsCuda benchmark::benchmark)
    set_target_properties(benchmarkBonds
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkBonds PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    set_property(TARGET benchmarkBonds PROPERTY CUDA_STANDARD 11)
    set_source_files_properties(benchmarkBonds.cpp PROPERTIES LANGUAGE CUDA)
endif()
