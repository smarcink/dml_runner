set(TARGET_NAME "cross_runner")

set(SOURCES_DIR "src")

set(TARGET_SOURCES
    ${SOURCES_DIR}/main.cpp
)

add_executable(${TARGET_NAME} ${TARGET_SOURCES})
target_compile_options(${TARGET_NAME} PUBLIC /W3 /MP)
target_link_libraries(${TARGET_NAME} PRIVATE common_lib)