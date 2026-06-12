set(TARGET llm_cli_impl)

add_library(${TARGET} cli.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../llm_server)
target_link_libraries(${TARGET} PUBLIC server_context llm_common lhm_models ${CMAKE_THREAD_LIBS_INIT})

# llm_cli executable

set(TARGET llm_cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llm_cli_impl fmt::fmt spdlog::spdlog gflags::gflags)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LHM_CLI_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
