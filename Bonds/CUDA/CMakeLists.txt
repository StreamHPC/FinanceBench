add_executable(bondsCUDA bondsEngine.cu)
set_target_properties(bondsCUDA
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Bonds"
)
