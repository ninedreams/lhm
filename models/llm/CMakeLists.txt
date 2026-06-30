lhm_add_compile_flags()

add_library(lhm_llm STATIC
    models.h
    delta_net_base.cpp
    qwen35.cpp
    qwen35moe.cpp
    deepseekv4.cpp
)

target_include_directories(lhm_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_llm PRIVATE lhm_params lhm_graph lhm_loader ggml)
