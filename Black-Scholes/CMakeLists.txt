add_library(blackScholesCpu blackScholesAnalyticEngineKernelsCpu.cpp)
add_executable(blackScholes blackScholesAnalyticEngine.cpp)
target_link_libraries(blackScholes blackScholesCpu)
set_target_properties(blackScholes
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(blackScholesCuda blackScholesAnalyticEngineKernels.cu)
    target_link_libraries(blackScholes blackScholesCuda)
    set_source_files_properties(blackScholesAnalyticEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(BUILD_BENCHMARK)
    add_executable(benchmarkBlackScholes benchmarkBlackScholes.cpp)
    target_link_libraries(benchmarkBlackScholes blackScholesCpu blackScholesCuda benchmark::benchmark)
    set_target_properties(benchmarkBlackScholes
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkBlackScholes PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    set_property(TARGET benchmarkBlackScholes PROPERTY CUDA_STANDARD 11)
    set_source_files_properties(benchmarkBlackScholes.cpp PROPERTIES LANGUAGE CUDA)
endif()
