add_executable(monteCarloCUDA monteCarloEngine.cu)
set_target_properties(monteCarloCUDA
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Monte-Carlo"
)
