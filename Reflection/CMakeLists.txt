# Header-only. REFLECTABLE relies on __VA_OPT__, which MSVC only expands
# correctly with the conforming preprocessor.
add_library(reflection INTERFACE)
add_library(InferenceEngine::Reflection ALIAS reflection)

target_include_directories(reflection INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

if(MSVC)
    target_compile_options(reflection INTERFACE /Zc:preprocessor)
endif()

target_sources(reflection INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Reflection.h
)
