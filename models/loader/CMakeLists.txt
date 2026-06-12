lhm_add_compile_flags()

add_library(lhm_loader STATIC
    lhm_mmap.cpp
    lhm_mmap.h
    lhm_model-loader.cpp
    lhm_model-loader.h
    lhm_model-saver.cpp
    lhm_model-saver.h
)

target_include_directories(lhm_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lhm_loader PUBLIC ${CMAKE_SOURCE_DIR}/models)
target_include_directories(lhm_loader PUBLIC ${CMAKE_SOURCE_DIR}/lhm)
target_compile_features(lhm_loader PRIVATE cxx_std_17) # don't bump

target_link_libraries(lhm_loader PUBLIC lhm_params ggml)

if(BUILD_SHARED_LIBS)
    set_target_properties(lhm_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
