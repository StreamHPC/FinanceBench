add_library(repoCpu repoKernelsCpu.cpp)
#add_executable(repo repoEngine.cpp)
#target_link_libraries(repo repoCpu)
#set_target_properties(repo
#  PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#)

if(BUILD_BENCHMARK)
    add_executable(benchmarkRepo benchmarkRepo.cpp)
    target_link_libraries(benchmarkRepo repoCpu benchmark::benchmark)
    set_target_properties(benchmarkRepo
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(benchmarkRepo PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endif()

if(BUILD_TEST)
    add_executable(testRepo testRepo.cpp)
    target_link_libraries(testRepo repoCpu ${GTEST_BOTH_LIBRARIES})
    target_include_directories(testRepo SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
  )
    set_target_properties(testRepo
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    add_test(NAME testRepo COMMAND testRepo)
endif()

if(BUILD_CUDA)
    add_library(repoCuda repoKernels.cu)
    #target_link_libraries(repo repoCuda)
    #set_source_files_properties(repoEngine.cpp PROPERTIES LANGUAGE CUDA)
    if(BUILD_BENCHMARK)
        target_link_libraries(benchmarkRepo repoCuda)
        set_property(TARGET benchmarkRepo PROPERTY CUDA_STANDARD 11)
        set_source_files_properties(benchmarkRepo.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    if(BUILD_TEST)
        target_link_libraries(testRepo repoCuda)
        set_property(TARGET testRepo PROPERTY CUDA_STANDARD 11)
        set_source_files_properties(testRepo.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endif()
