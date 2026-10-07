# Header-only for now: INTERFACE carries the include path to consumers
# without producing a library artifact.
add_library(tensor INTERFACE)
add_library(InferenceEngine::Tensor ALIAS tensor)

target_include_directories(tensor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(tensor INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES Tensor.h TensorCore.h Ops.h detail/Shape.h detail/CudaOps.h
)

if(IE_ENABLE_CUDA)
    target_link_libraries(tensor INTERFACE InferenceEngine::CUDA)
    target_compile_definitions(tensor INTERFACE IE_ENABLE_CUDA)
endif()
