add_library(ui INTERFACE)
add_library(InferenceEngine::UI ALIAS ui)

target_include_directories(ui INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(ui INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ChatUI.h
)
