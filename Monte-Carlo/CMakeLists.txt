add_library(monteCarloCpu monteCarloKernelsCpu.cpp)
add_executable(monteCarlo monteCarloEngine.cpp)
target_link_libraries(monteCarlo monteCarloCpu)
set_target_properties(monteCarlo
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(monteCarloCuda monteCarloKernels.cu)
    target_link_libraries(monteCarlo monteCarloCuda)
    set_source_files_properties(monteCarloEngine.cpp PROPERTIES LANGUAGE CUDA)
    set_property(TARGET monteCarlo PROPERTY CUDA_STANDARD 11)
    set_property(TARGET monteCarloCuda PROPERTY CUDA_STANDARD 11)
endif()

if(BUILD_BENCHMARK)
    add_executable(benchmarkMonteCarlo benchmarkMonteCarlo.cpp)
    target_link_libraries(benchmarkMonteCarlo monteCarloCpu monteCarloCuda benchmark::benchmark)
    set_target_properties(benchmarkMonteCarlo
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkMonteCarlo PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    set_property(TARGET benchmarkMonteCarlo PROPERTY CUDA_STANDARD 11)
    set_source_files_properties(benchmarkMonteCarlo.cpp PROPERTIES LANGUAGE CUDA)
endif()
