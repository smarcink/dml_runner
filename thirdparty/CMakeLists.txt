cmake_policy(SET CMP0077 NEW)

#d3d12x
add_library(d3d12x INTERFACE)
target_include_directories(d3d12x INTERFACE "d3dx12")

#dmlx
add_library(dmlx INTERFACE)
target_include_directories(dmlx INTERFACE "dmlx")

#one dnll
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # so we can set options here
set(ONEDNN_LIBRARY_TYPE SHARED)
set(ONEDNN_CPU_RUNTIME NONE)
set(ONEDNN_GPU_RUNTIME IUMD)
set(ONEDNN_BUILD_WITH_MOCK_OPENCL OFF)  # for driver integration this has to be "ON" - no OCL depdencies allowed!
set(ONEDNN_BUILD_EXAMPLES OFF)
set(ONEDNN_BUILD_TESTS OFF)
set(ONEDNN_ENABLE_WORKLOAD INFERENCE)
set(ONEDNN_BUILD_GRAPH OFF)
set(DNNL_EXPERIMENTAL_PROFILING ON)
add_subdirectory(drivers.gpu.compute.ai.dnnlpluginnext)

#igdext
add_library(cfgmgr UNKNOWN IMPORTED)
set_property(TARGET cfgmgr PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/igdext/lib/cfgmgr32.lib)

add_library(setupapi UNKNOWN IMPORTED)
set_property(TARGET setupapi PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/igdext/lib/SetupAPI.lib)

add_library(ShLwApi UNKNOWN IMPORTED)
set_property(TARGET ShLwApi PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/igdext/lib/ShLwApi.lib)

add_library(igdext STATIC IMPORTED GLOBAL)
set_property(TARGET igdext PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/igdext/include)
set_property(TARGET igdext PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/igdext/lib/igdext64.lib)
set_property(TARGET igdext PROPERTY INTERFACE_LINK_LIBRARIES cfgmgr setupapi ShLwApi)
  
#dml
add_library(dml SHARED IMPORTED GLOBAL)
set_property(TARGET dml PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dml/include")
set_property(TARGET dml PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/dml/bin/x64-win/DirectML.dll")
set_property(TARGET dml PROPERTY IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/dml/bin/x64-win/DirectML.lib")

#cli
add_subdirectory(CLI11)

#gtests
add_subdirectory(googletest)