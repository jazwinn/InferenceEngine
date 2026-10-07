file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

file(GLOB CUDA_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_kernels STATIC ${CUDA_SOURCES})
add_library(InferenceEngine::CUDA ALIAS cuda_kernels)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES native
)

target_link_libraries(cuda_kernels
    PUBLIC InferenceEngine::Tensor
    PRIVATE CUDA::cudart_static
)

target_include_directories(cuda_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_sources(cuda_kernels PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ${CUDA_HEADERS}
)
