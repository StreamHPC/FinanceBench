add_executable(blackScholesOpenCL blackScholesAnalyticEngine.cpp)
target_link_libraries(blackScholesOpenCL
  PUBLIC
    ${OpenCL_LIBRARIES}
)
target_include_directories(blackScholesOpenCL
  PUBLIC
    $<BUILD_INTERFACE:${OpenCL_INCLUDE_DIRS}>
)
set_target_properties(blackScholesOpenCL
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Black-Scholes"
)

add_custom_command(TARGET blackScholesOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/blackScholesAnalyticEngineKernels.cl ${CMAKE_BINARY_DIR}/Black-Scholes/blackScholesAnalyticEngineKernels.cl)
add_custom_command(TARGET blackScholesOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/blackScholesAnalyticEngineKernels.h ${CMAKE_BINARY_DIR}/Black-Scholes/blackScholesAnalyticEngineKernels.h)
add_custom_command(TARGET blackScholesOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/errorFunctConsts.h ${CMAKE_BINARY_DIR}/Black-Scholes/errorFunctConsts.h)
add_custom_command(TARGET blackScholesOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/blackScholesStructs.h ${CMAKE_BINARY_DIR}/Black-Scholes/blackScholesStructs.h)
add_custom_command(TARGET blackScholesOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/blackScholesAnalyticEngineStructs.h ${CMAKE_BINARY_DIR}/Black-Scholes/blackScholesAnalyticEngineStructs.h)
