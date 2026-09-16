# Header-only, like Math: INTERFACE carries the include path and the dependency
# on the tensor library to consumers.
add_library(neuralnet INTERFACE)
add_library(InferenceEngine::NeuralNet ALIAS neuralnet)

target_link_libraries(neuralnet INTERFACE InferenceEngine::Math)

target_include_directories(neuralnet INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(neuralnet INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Module.h Linear.h Activation.h
)
