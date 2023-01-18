#d3d12x
add_library(d3d12x INTERFACE)
target_include_directories(d3d12x INTERFACE "d3dx12")

#dmlx
add_library(dmlx INTERFACE)
target_include_directories(dmlx INTERFACE "dmlx")

#one dnll
set(ONEDNN_LIBRARY_TYPE SHARED)
set(ONEDNN_CPU_RUNTIME NONE)
set(ONEDNN_GPU_RUNTIME OCL)
set(ONEDNN_BUILD_EXAMPLES OFF)
set(ONEDNN_BUILD_TESTS OFF)
set(ONEDNN_ENABLE_WORKLOAD INFERENCE)
add_subdirectory(oneDNN)

#dml
add_library(dml SHARED IMPORTED GLOBAL)
set_property(TARGET dml PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dml/include")
set_property(TARGET dml PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/dml/bin/x64-win/DirectML.dll")
set_property(TARGET dml PROPERTY IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/dml/bin/x64-win/DirectML.lib")

#cli
add_subdirectory(CLI11)