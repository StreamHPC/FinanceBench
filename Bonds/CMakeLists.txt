add_library(bondsCpu bondsKernelsCpu.cpp)
add_executable(bonds bondsEngine.cpp)
target_link_libraries(bonds bondsCpu)
set_target_properties(bonds
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(bondsCuda bondsKernelsGpu.cu)
    target_link_libraries(bonds bondsCuda)
    set_source_files_properties(bondsEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()
