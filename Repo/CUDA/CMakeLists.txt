add_executable(repoCUDA repoEngine.cu)
set_target_properties(repoCUDA
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Repo"
)
