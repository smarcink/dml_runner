set(TARGET_NAME "libdml")

set(API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DML_API_SOURCES 
    ${API_DIR}/dml_types.hpp
    ${API_DIR}/dml_convolution.hpp
)  

set(SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DML_SOURCES 
    ${SOURCES_DIR}/dml_convolution.cpp
)    
    
set(CONVOLUTION_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/convolution)
set(DML_CONV_IMPL_SOURCES 
    ${CONVOLUTION_SOURCES_DIR}/convolution_impl.h
)      

set(ALL_SOURCES ${DML_API_SOURCES} ${DML_SOURCES} ${DML_CONV_IMPL_SOURCES})
    
message(STATUS ${ALL_SOURCES})
add_library(${TARGET_NAME} STATIC ${ALL_SOURCES})
target_include_directories(${TARGET_NAME} PUBLIC ${API_DIR})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_SOURCES})