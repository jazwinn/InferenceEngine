# Header-only. REFLECTABLE relies on __VA_OPT__, which MSVC only expands
# correctly with the conforming preprocessor. clang-cl also sets MSVC but its
# preprocessor is already conforming, so it doesn't need the flag.
add_library(reflection INTERFACE)
add_library(InferenceEngine::Reflection ALIAS reflection)

target_include_directories(reflection INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(reflection INTERFACE /Zc:preprocessor)
endif()

target_sources(reflection INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Reflection.h
)
