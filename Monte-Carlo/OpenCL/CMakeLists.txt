add_executable(monteCarloOpenCL monteCarloEngine.cpp)
target_link_libraries(monteCarloOpenCL
  PUBLIC
    ${OpenCL_LIBRARIES}
)
target_include_directories(monteCarloOpenCL
  PUBLIC
    $<BUILD_INTERFACE:${OpenCL_INCLUDE_DIRS}>
)
set_target_properties(monteCarloOpenCL
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Monte-Carlo"
)

add_custom_command(TARGET monteCarloOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/monteCarloKernels.cl ${CMAKE_BINARY_DIR}/Monte-Carlo/monteCarloKernels.cl)
add_custom_command(TARGET monteCarloOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/monteCarloKernels.h ${CMAKE_BINARY_DIR}/Monte-Carlo/monteCarloKernels.h)
add_custom_command(TARGET monteCarloOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/monteCarloConstants.h ${CMAKE_BINARY_DIR}/Monte-Carlo/monteCarloConstants.h)
add_custom_command(TARGET monteCarloOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/monteCarloStructs.h ${CMAKE_BINARY_DIR}/Monte-Carlo/monteCarloStructs.h)
add_custom_command(TARGET monteCarloOpenCL POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/mt19937.h ${CMAKE_BINARY_DIR}/Monte-Carlo/mt19937.h)
