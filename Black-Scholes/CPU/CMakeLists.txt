add_executable(blackScholesCPU blackScholesAnalyticEngine.cpp)
set_target_properties(blackScholesCPU
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Black-Scholes"
)
