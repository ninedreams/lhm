lhm_add_compile_flags()

add_library(lhm_loader STATIC
    lhm_mmap.cpp
    lhm_mmap.h
    lhm_model_loader.cpp
    lhm_model_loader.h
    lhm_model_saver.cpp
    lhm_model_saver.h
)

target_include_directories(lhm_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_loader PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_loader PUBLIC ${CMAKE_SOURCE_DIR}/lhm)
target_compile_features(lhm_loader PRIVATE cxx_std_17) # don't bump

target_link_libraries(lhm_loader PUBLIC lhm_params ggml)
