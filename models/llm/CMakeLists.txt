lhm_add_compile_flags()

file(GLOB LHM_LLM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(lhm_llm STATIC
    models.h
    ${LHM_LLM_SOURCES}
)

target_include_directories(lhm_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_llm PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_llm PRIVATE lhm_params lhm_loader ggml)
