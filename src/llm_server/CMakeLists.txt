include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server_context containing the core server logic, used by llm_server and CLI
set(TARGET server_context)

add_library(${TARGET} STATIC
    server_chat.cpp
    server_chat.h
    server_task.cpp
    server_task.h
    server_queue.cpp
    server_queue.h
    server_common.cpp
    server_common.h
    server_context.cpp
    server_context.h
    server_tools.cpp
    server_tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llm_common lhm_models mtmd ${CMAKE_THREAD_LIBS_INIT})

# llm_server-impl: server logic, reusable by app
set(TARGET llm_server_impl)

add_library(${TARGET}
    server.cpp
    server_http.cpp
    server_http.h
    server_models.cpp
    server_models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server_context httplib::httplib ${CMAKE_THREAD_LIBS_INIT})

# llm_server executable

set(TARGET llm_server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llm_server_impl fmt::fmt spdlog::spdlog gflags::gflags)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
