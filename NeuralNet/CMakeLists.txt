find_package(nlohmann_json CONFIG REQUIRED)

add_library(neuralnet INTERFACE)
add_library(InferenceEngine::NeuralNet ALIAS neuralnet)

target_link_libraries(neuralnet INTERFACE InferenceEngine::Tensor nlohmann_json::nlohmann_json)

target_include_directories(neuralnet INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

file(GLOB_RECURSE NEURALNET_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(neuralnet INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ${NEURALNET_HEADERS}
)
