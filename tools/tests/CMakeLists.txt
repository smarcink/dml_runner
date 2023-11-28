cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME "tests")
project(${PROJECT_NAME})

# add main
set(MAIN_LABEL "")
file(GLOB MAIN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

# add the test cases
set(TEST_CASES_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/test_cases")
set(TEST_CASES_LABEL    "test cases")
file(GLOB TEST_CASES_SOURCES
    "${TEST_CASES_DIR}/*.h"
    "${TEST_CASES_DIR}/*.hpp"
    "${TEST_CASES_DIR}/*.cpp")

# add the gtest
set(GTEST_DIR           "${CMAKE_CURRENT_SOURCE_DIR}/googletest-fused")
set(GTEST_LABEL         "google test framework")
file(GLOB GTEST_SOURCES "${GTEST_DIR}/*.cc")


set(ALL_SOURCES
    ${MAIN_SOURCES}
    ${TEST_CASES_SOURCES}
    ${GTEST_SOURCES}
  )

source_group("${MAIN_LABEL}"        FILES ${MAIN_SOURCES})
source_group("${TEST_CASES_LABEL}"  FILES ${TEST_CASES_SOURCES})
source_group("${TEST_UTILS_LABEL}"  FILES ${TEST_UTILS_SOURCES})
source_group("${GTEST_LABEL}"       FILES ${TEST_SOURCES})

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${GTEST_DIR}"
  )

# Tests executable.
add_executable(${PROJECT_NAME} ${ALL_SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE common_lib)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W1)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wextra -pedantic)
endif()