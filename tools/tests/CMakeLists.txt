cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME "tests")
project(${PROJECT_NAME})

# add main
set(MAIN_LABEL "")
set(MAIN_SOURCES
    main.cpp
)

# add the test cases
set(TEST_CASES_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/test_cases")
set(TEST_CASES_LABEL    "test cases")
set(TEST_CASES_SOURCES
    ${TEST_CASES_DIR}/test_gemm.cpp
    ${TEST_CASES_DIR}/test_gemm_parametrized.cpp
    ${TEST_CASES_DIR}/test_conv.cpp
    ${TEST_CASES_DIR}/test_conv_parametrized.cpp
    ${TEST_CASES_DIR}/test_api.cpp
    
    ${TEST_CASES_DIR}/utils.h
)


set(ALL_SOURCES
    ${MAIN_SOURCES}
    ${TEST_CASES_SOURCES}
  )

source_group("${MAIN_LABEL}"        FILES ${MAIN_SOURCES})
source_group("${TEST_CASES_LABEL}"  FILES ${TEST_CASES_SOURCES})
source_group("${TEST_UTILS_LABEL}"  FILES ${TEST_UTILS_SOURCES})

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

# Tests executable.
add_executable(${PROJECT_NAME} ${ALL_SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE common_lib gtest_main)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W1 /MP)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wextra -pedantic)
endif()