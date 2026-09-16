# Header-only for now: INTERFACE carries the include path to consumers
# without producing a library artifact.
add_library(math INTERFACE)
add_library(InferenceEngine::Math ALIAS math)

target_include_directories(math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(math INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Tensor.h
)
