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
    
    ${SOURCES_DIR}/impl/gpu_context.cpp
    ${SOURCES_DIR}/impl/gpu_context.h
    ${SOURCES_DIR}/impl/model.cpp
    ${SOURCES_DIR}/impl/model.h
	${SOURCES_DIR}/impl/tensor.h
	${SOURCES_DIR}/impl/node.h
    
	${SOURCES_DIR}/impl/nodes/activation.h
	${SOURCES_DIR}/impl/nodes/activation.cpp
	${SOURCES_DIR}/impl/nodes/matmul.h
	${SOURCES_DIR}/impl/nodes/matmul.cpp
	${SOURCES_DIR}/impl/nodes/port.h
)

add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})
target_include_directories(${TARGET_NAME} PUBLIC ${HEADERS_DIR})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME} PRIVATE /W3)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_SOURCES})
