set(TARGET llm_cli_impl)

add_library(${TARGET} cli.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../llm_server)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR} common)
target_link_libraries(${TARGET} PUBLIC
    common
    server_context
    lhm_models 
    lhm_params
    lhm_kvcache
    lhm_memory
    lhm_loader
    lhm_llm
    lhm_chat
    fmt::fmt
    spdlog::spdlog
    gflags
    ggml
    ${CMAKE_THREAD_LIBS_INIT})

# llm_cli executable

set(TARGET llm_cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llm_cli_impl)

install(TARGETS ${TARGET} RUNTIME)
