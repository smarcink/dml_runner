set(TARGET_NAME "inference_engine")

set(HEADERS_DIR "include")
set(SOURCES_DIR "src")


set(TARGET_SOURCES
    ${HEADERS_DIR}/inference_engine.h
    ${HEADERS_DIR}/inference_engine_export.h
    ${HEADERS_DIR}/inference_engine_operators.h
    ${HEADERS_DIR}/inference_engine_tensor.h
    ${HEADERS_DIR}/inference_engine_model.h
    
    ${SOURCES_DIR}/inference_engine.cpp
    ${SOURCES_DIR}/inference_engine_operators.cpp
    ${SOURCES_DIR}/inference_engine_model.cpp
    
    ${SOURCES_DIR}/impl/model.cpp
    ${SOURCES_DIR}/impl/model.h
)

add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE d3d12 dxgi dxguid d3d12x)
target_include_directories(${TARGET_NAME} PUBLIC ${HEADERS_DIR})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME} PRIVATE /W3)

