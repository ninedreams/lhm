lhm_add_compile_flags()

add_library(lhm_kvcache STATIC
    lhm_kv_cache.cpp
    lhm_kv_cache.h
    lhm_kv_cache_dsv4.cpp
    lhm_kv_cache_dsv4.h
    lhm_kv_cache_dsa.cpp
    lhm_kv_cache_dsa.h
    lhm_kv_cache_iswa.cpp
    lhm_kv_cache_iswa.h
    lhm_kv_cells.h
)

if (LHM_ENABLE_MOONCAKE)
    target_sources(lhm_kvcache PRIVATE
        lhm_kv_cache_mooncake.cpp
        lhm_kv_cache_mooncake.h
    )
    target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include)
    target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator)
    target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator/include)
    target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Mooncake/mooncake-store/include/cachelib_memory_allocator/fake_include)
    target_link_libraries(lhm_kvcache PUBLIC mooncake_store)
endif()

target_include_directories(lhm_kvcache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_kvcache PRIVATE lhm_params lhm_memory ggml gflags)

