lhm_add_compile_flags()

add_library(lhm_params STATIC
    lhm_cparams.cpp
    lhm_cparams.h
    lhm_hparams.cpp
    lhm_hparams.h
)

target_include_directories(lhm_params PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_params PUBLIC ${CMAKE_SOURCE_DIR}/lhm)

target_link_libraries(lhm_params PRIVATE common ggml)
