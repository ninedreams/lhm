find_package(Threads REQUIRED)

lhm_add_compile_flags()

#
# lhm_common
#
set(TARGET lhm_common)

add_library(${TARGET}
    base64.hpp
    chat_auto_parser_generator.cpp
    chat_auto_parser_helpers.cpp
    chat_auto_parser.h
    chat_diff_analyzer.cpp
    chat_peg_parser.cpp
    chat_peg_parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    debug.cpp
    debug.h
    fit.cpp
    fit.h
    http.h
    imatrix_loader.cpp
    imatrix_loader.h
    json_partial.cpp
    json_partial.h
    json_schema_to_grammar.cpp
    llguidance.cpp
    lhm_assert.h
    log.cpp
    log.h
    ngram_cache.cpp
    ngram_cache.h
    ngram_map.cpp
    ngram_map.h
    ngram_mod.cpp
    ngram_mod.h
    peg_parser.cpp
    peg_parser.h
    regex_partial.cpp
    reasoning_budget.cpp
    reasoning_budget.h
    regex_partial.h
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    unicode.cpp
    unicode.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
)

target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/fmt/include)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/spdlog/include)

target_link_libraries(${TARGET} PRIVATE httplib::httplib gflags::gflags fmt::fmt spdlog::spdlog RapidJSON)

if(LHM_LLGUIDANCE)
    message("Not support LHM_LLGUIDANCE")
endif()
