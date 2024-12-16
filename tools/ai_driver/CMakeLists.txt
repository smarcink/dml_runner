set(TARGET_NAME "ai_driver")

set(HEADERS_DIR "include")
set(SOURCES_DIR "src")


set(TARGET_SOURCES
    ${HEADERS_DIR}/ai_driver.h
    ${HEADERS_DIR}/ai_driver_export.h
    ${HEADERS_DIR}/ai_driver_operators.h
    ${HEADERS_DIR}/ai_driver_tensor.h
    ${HEADERS_DIR}/ai_driver_model.h
    
    ${HEADERS_DIR}/ai_driver.hpp
    ${HEADERS_DIR}/ai_driver_operators.hpp
    ${HEADERS_DIR}/ai_driver_tensor.hpp
    ${HEADERS_DIR}/ai_driver_model.hpp
    
    ${SOURCES_DIR}/ai_driver.cpp
    ${SOURCES_DIR}/ai_driver_operators.cpp
    ${SOURCES_DIR}/ai_driver_model.cpp
    
    ${SOURCES_DIR}/impl/gpu_context.cpp
    ${SOURCES_DIR}/impl/gpu_context.h
    ${SOURCES_DIR}/impl/gpu_visitor.h
    ${SOURCES_DIR}/impl/model.cpp
    ${SOURCES_DIR}/impl/model.h
	${SOURCES_DIR}/impl/node.h
	${SOURCES_DIR}/impl/node.cpp
    
	${SOURCES_DIR}/impl/nodes/activation.h
	${SOURCES_DIR}/impl/nodes/activation.cpp
	${SOURCES_DIR}/impl/nodes/matmul.h
	${SOURCES_DIR}/impl/nodes/matmul.cpp
	${SOURCES_DIR}/impl/nodes/port.h
	${SOURCES_DIR}/impl/nodes/port.cpp
    ${SOURCES_DIR}/impl/nodes/constant_port.h
	${SOURCES_DIR}/impl/nodes/constant_port.cpp
	${SOURCES_DIR}/impl/nodes/elementwise.h
	${SOURCES_DIR}/impl/nodes/elementwise.cpp
	${SOURCES_DIR}/impl/nodes/convolution.h
	${SOURCES_DIR}/impl/nodes/convolution.cpp
)

add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})
target_include_directories(${TARGET_NAME} PUBLIC ${HEADERS_DIR})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME} PRIVATE /W3)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_SOURCES})
