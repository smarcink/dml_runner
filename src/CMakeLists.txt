set(TARGET_NAME "tester")

set(TARGET_SOURCES
    main.cpp
    dx12_utils.h
    dml_base_node.h
    layers_utils.h
    dnnl_utils.h
    gemm.h
    conv.h
    conv.cpp
    softmax.h
    softmax.cpp
    mvn.h
    mvn.cpp
)

add_executable(${TARGET_NAME} ${TARGET_SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE dml d3d12 dxgi dxguid d3d12x dmlx dnnl CLI11::CLI11 igdext)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME} PRIVATE /W3)

