find_package(nlohmann_json CONFIG REQUIRED)

add_library(neuralnet INTERFACE)
add_library(InferenceEngine::NeuralNet ALIAS neuralnet)

target_link_libraries(neuralnet INTERFACE InferenceEngine::Math nlohmann_json::nlohmann_json)

target_include_directories(neuralnet INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(neuralnet INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Module.h Linear.h Activation.h LayerNorm.h RMSNorm.h Transformer/MultiHeadAttention.h Transformer/Transformer.h Transformer/TransformerConfig.h Sequential.h
)
