add_executable(monteCarloCPU monteCarloEngine.cpp)
set_target_properties(monteCarloCPU
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Monte-Carlo"
)
