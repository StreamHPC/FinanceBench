add_library(blackScholesCpu blackScholesAnalyticEngineKernelsCpu.cpp)
add_executable(blackScholes blackScholesAnalyticEngine.cpp)
target_link_libraries(blackScholes blackScholesCpu)
set_target_properties(blackScholes
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(blackScholesCuda blackScholesAnalyticEngineKernels.cu)
    target_link_libraries(blackScholes blackScholesCuda)
    set_source_files_properties(blackScholesAnalyticEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()
