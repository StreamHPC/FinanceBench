add_library(repoCpu repoKernelsCpu.cpp)
add_executable(repo repoEngine.cpp)
target_link_libraries(repo repoCpu)
set_target_properties(repo
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(repoCuda repoKernels.cu)
    target_link_libraries(repo repoCuda)
    set_source_files_properties(repoEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()
