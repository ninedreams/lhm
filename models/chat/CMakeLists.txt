lhm_add_compile_flags()

add_library(lhm_chat STATIC
    chat.cpp
    chat.h
    chat_auto_parser.h
    chat_auto_parser_generator.cpp
    chat_auto_parser_helpers.cpp
    chat_auto_parser_helpers.h
    chat_diff_analyzer.cpp
    chat_peg_parser.cpp
    chat_peg_parser.h
)

target_include_directories(lhm_chat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_chat PUBLIC ${CMAKE_SOURCE_DIR}/common)
target_include_directories(lhm_chat PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_chat PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_chat PRIVATE common ggml lhm_models)
