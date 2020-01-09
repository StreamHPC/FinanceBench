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
