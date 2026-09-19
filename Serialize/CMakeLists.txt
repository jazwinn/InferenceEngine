find_package(nlohmann_json CONFIG REQUIRED)

# Header-only JSON loading for types made reflectable with REFLECTABLE(...),
# and reading tensors from .safetensors files.
add_library(serialize INTERFACE)
add_library(InferenceEngine::Serialize ALIAS serialize)

target_link_libraries(serialize INTERFACE InferenceEngine::Math InferenceEngine::Reflection nlohmann_json::nlohmann_json)

target_include_directories(serialize INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(serialize INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Serialize.h Safetensors.h
)
