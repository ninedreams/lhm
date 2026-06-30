lhm_add_compile_flags()

add_library(lhm_graph STATIC
    lhm_graph_dsv4.cpp
    lhm_graph_dsv4.h
    lhm_graph.cpp
    lhm_graph.h
)

target_include_directories(lhm_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_graph PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_graph PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_graph PRIVATE lhm_params lhm_memory lhm_kvcache ggml)
