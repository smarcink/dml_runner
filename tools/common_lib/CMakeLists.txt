set(TARGET_NAME "common_lib")

set(SOURCES_DIR "src")

set(TARGET_SOURCES
    ${SOURCES_DIR}/dx12_utils.h
    ${SOURCES_DIR}/dml_base_node.h
    ${SOURCES_DIR}/layers_utils.h
    ${SOURCES_DIR}/dnnl_utils.h
    ${SOURCES_DIR}/tensor_shape.h
    ${SOURCES_DIR}/gemm.h
    ${SOURCES_DIR}/gemm.cpp
    ${SOURCES_DIR}/conv.h
    ${SOURCES_DIR}/conv.cpp
    ${SOURCES_DIR}/softmax.h
    ${SOURCES_DIR}/softmax.cpp
    ${SOURCES_DIR}/mvn.h
    ${SOURCES_DIR}/mvn.cpp
	${SOURCES_DIR}/mha.h
    ${SOURCES_DIR}/memory_bandwidth.h
    ${SOURCES_DIR}/iumd_d3d12_impl.h
    ${SOURCES_DIR}/iumd_d3d12_impl.cpp
)

add_library(${TARGET_NAME} STATIC ${TARGET_SOURCES})
target_link_libraries(${TARGET_NAME} PUBLIC dml d3d12 dxgi dxguid d3d12x dmlx dnnl CLI11::CLI11 igdext)
target_include_directories(${TARGET_NAME} PUBLIC "src")
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)
target_compile_options(${TARGET_NAME} PUBLIC /W3)
target_compile_definitions(${TARGET_NAME} PUBLIC DML_TARGET_VERSION_USE_LATEST)

