lhm_add_compile_flags()

#
# sub-directory libraries
#
add_subdirectory(chat)
add_subdirectory(graph)
add_subdirectory(kvcache)
add_subdirectory(llm)
add_subdirectory(loader)
add_subdirectory(memory)
add_subdirectory(params)

#
# main library: lhm
#
set(TARGET lhm_models)

add_library(${TARGET} STATIC
    ../lhm/lhm.h
    ../lhm/lhm_cpp.h
    lhm.cpp
    lhm_adapter.cpp
    lhm_adapter.h
    lhm_arch.cpp
    lhm_arch.h
    lhm_batch.cpp
    lhm_batch.h
    lhm_chat.cpp
    lhm_chat.h
    lhm_context.cpp
    lhm_context.h
    lhm_ext.h
    lhm_grammar.cpp
    lhm_grammar.h
    lhm_impl.cpp
    lhm_impl.h
    lhm_io.cpp
    lhm_io.h
    lhm_model.cpp
    lhm_model.h
    lhm_quant.cpp
    lhm_quant.h
    lhm_sampler.cpp
    lhm_sampler.h
    lhm_vocab.cpp
    lhm_vocab.h
    unicode_data.cpp
    unicode_data.h
    unicode.cpp
    unicode.h
)

target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/lhm)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/common)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/fmt/include)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/spdlog/include)
if (LHM_ENABLE_MOONCAKE)
    target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include)
    target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator)
    target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator/include)
    target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator/fake_include)
endif()

target_link_libraries(${TARGET} PRIVATE
    common
    ggml
    gflags
    fmt::fmt
    spdlog::spdlog
)

target_link_libraries(${TARGET} PUBLIC
    lhm_params
    lhm_kvcache
    lhm_memory
    lhm_loader
    lhm_llm
    lhm_graph
    lhm_chat
)
