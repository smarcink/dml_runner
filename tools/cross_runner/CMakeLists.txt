set(TARGET_NAME "cross_runner")

set(SOURCES_DIR "src")

set(TARGET_SOURCES
    ${SOURCES_DIR}/main.cpp
    ${SOURCES_DIR}/dx12_utils.h
    ${SOURCES_DIR}/dml_base_node.h
    ${SOURCES_DIR}/layers_utils.h
    ${SOURCES_DIR}/dnnl_utils.h
    ${SOURCES_DIR}/gemm.h
    ${SOURCES_DIR}/conv.h
    ${SOURCES_DIR}/conv.cpp
    ${SOURCES_DIR}/softmax.h
    ${SOURCES_DIR}/softmax.cpp
    ${SOURCES_DIR}/mvn.h
    ${SOURCES_DIR}/mvn.cpp
    ${SOURCES_DIR}/memory_bandwidth.h
    ${SOURCES_DIR}/iumd_d3d12_impl.h
)

add_executable(${TARGET_NAME} ${TARGET_SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE dml d3d12 dxgi dxguid d3d12x dmlx dnnl CLI11::CLI11 igdext libdml)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME} PRIVATE /W3)

