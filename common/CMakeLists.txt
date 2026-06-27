find_package(Threads REQUIRED)

lhm_add_compile_flags()

#
# common
#
set(TARGET common)

add_library(${TARGET}
    base64.hpp
    common_params.cpp
    common_params.h
    common.cpp
    common.h
    config.h
    config.cpp
    console.cpp
    console.h
    debug.cpp
    debug.h
    # fit.cpp
    # fit.h
    http.h
    imatrix_loader.cpp
    imatrix_loader.h
    json_partial.cpp
    json_partial.h
    json_schema_to_grammar.cpp
    lassert.h
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
    # speculative.cpp
    # speculative.h
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

target_link_libraries(${TARGET} PRIVATE httplib::httplib gflags::gflags fmt::fmt spdlog::spdlog RapidJSON ggml)

if(LHM_LLGUIDANCE)
    message("Not support LHM_LLGUIDANCE")
endif()
