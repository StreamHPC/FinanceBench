add_executable(blackScholesCUDA blackScholesAnalyticEngine.cu)
set_target_properties(blackScholesCUDA
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Black-Scholes"
)
