add_executable(bondsCPU bondsEngine.cpp)
set_target_properties(bondsCPU
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Bonds"
)
