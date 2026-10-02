find_package(nlohmann_json CONFIG REQUIRED)
find_package(pcre2 CONFIG REQUIRED)

add_library(models INTERFACE)
add_library(InferenceEngine::Models ALIAS models)

target_link_libraries(models INTERFACE
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
    nlohmann_json::nlohmann_json
    PCRE2::8BIT
)

target_include_directories(models INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

file(GLOB_RECURSE MODELS_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(models INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ${MODELS_HEADERS}
)
