set(TARGET llm_cli_impl)

add_library(${TARGET} cli.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../llm_server)
target_link_libraries(${TARGET} PUBLIC server_context common lhm_models ${CMAKE_THREAD_LIBS_INIT})

# llm_cli executable

set(TARGET llm_cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llm_cli_impl fmt::fmt spdlog::spdlog gflags::gflags ggml)

install(TARGETS ${TARGET} RUNTIME)
