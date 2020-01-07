if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(monteCarloCpu monteCarloKernelsCpu.cpp)
add_executable(monteCarlo monteCarloEngine.cpp)
target_link_libraries(monteCarlo monteCarloCpu)
set_target_properties(monteCarlo
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(BUILD_CUDA)
    add_library(monteCarloCuda monteCarloKernels.cu)
    target_link_libraries(monteCarlo monteCarloCuda)
    set_source_files_properties(monteCarloEngine.cpp PROPERTIES LANGUAGE CUDA)
endif()
