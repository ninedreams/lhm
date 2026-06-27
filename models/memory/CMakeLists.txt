lhm_add_compile_flags()

add_library(lhm_memory STATIC
    lhm_memory.cpp
    lhm_memory.h
    lhm_memory_hybrid.cpp
    lhm_memory_hybrid.h
    lhm_memory_hybrid_iswa.cpp
    lhm_memory_hybrid_iswa.h
    lhm_memory_recurrent.cpp
    lhm_memory_recurrent.h
)

target_include_directories(lhm_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_memory PUBLIC ${CMAKE_SOURCE_DIR}/lhm)
target_compile_features(lhm_memory PRIVATE cxx_std_17) # don't bump

target_link_libraries(lhm_memory PUBLIC lhm_params ggml)
