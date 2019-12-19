add_executable(repoCPU repoEngine.cpp)
set_target_properties(repoCPU
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Repo"
)
