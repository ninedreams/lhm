lhm_add_compile_flags()

add_library(lhm_llm STATIC
    base_models.h
    models.h
    delta_net_base.cpp
    qwen35.cpp
    qwen35.h
    qwen35moe.cpp
    qwen35moe.h
    deepseekv4.cpp
    deepseekv4.h
)

target_include_directories(lhm_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_llm PRIVATE lhm_params lhm_graph lhm_loader ggml)
