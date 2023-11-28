cmake_minimum_required(VERSION 3.23)  # actually older versions should also work ToDo: find lowest supported version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
cmake_policy(SET CMP0077 NEW)
set(PROJECT_NAME "libdml")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(${PROJECT_NAME})

add_subdirectory(thirdparty)
add_subdirectory(tools)