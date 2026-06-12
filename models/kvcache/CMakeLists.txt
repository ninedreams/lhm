lhm_add_compile_flags()

add_library(lhm_kvcache STATIC
    lhm_kv_cache.cpp
    lhm_kv_cache.h
    lhm_kv_cache_dsa.cpp
    lhm_kv_cache_dsa.h
    lhm_kv_cache_iswa.cpp
    lhm_kv_cache_iswa.h
    lhm_kv_cells.h
)

target_include_directories(lhm_kvcache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_kvcache PUBLIC ${CMAKE_SOURCE_DIR}/lhm)
target_compile_features(lhm_kvcache PRIVATE cxx_std_17) # don't bump

target_link_libraries(lhm_kvcache PUBLIC lhm_params lhm_memory ggml)
