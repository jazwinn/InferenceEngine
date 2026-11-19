cmake_minimum_required(VERSION 3.24)

# Must run before project(): the toolchain governs dependency resolution.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/vcpkg.cmake)

option(IE_ENABLE_CUDA "Build the CUDA backend" ON)
set(IE_CUDA_ARCHITECTURES "native" CACHE STRING "GPU architectures for the CUDA kernels (e.g. native, or 75;80;86;89;90)")
set(IE_PACKAGE_VERSION "0.0.0" CACHE STRING "Version recorded in the installed CMake package (set by the release workflow)")

project(InferenceEngine LANGUAGES CXX)

if(IE_ENABLE_CUDA)
    enable_language(CUDA)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single-config generators" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(IE_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC)
    # /bigobj: the header-only templates give main.cpp more sections than a
    # default object file allows in Debug builds (MSVC error C1128).
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive-> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

add_subdirectory(Generation)
add_subdirectory(Tensor)
if(IE_ENABLE_CUDA)
    add_subdirectory(CUDA)
endif()
add_subdirectory(NeuralNet)
add_subdirectory(Reflection)
add_subdirectory(Serialize)
add_subdirectory(Models)
add_subdirectory(UI)

# `cmake --install` produces a relocatable package (headers, the CUDA kernel
# library and InferenceEngineConfig.cmake) for projects that consume a
# release instead of the source. See cmake/Install.cmake.
option(IE_INSTALL "Generate install rules for the InferenceEngine package" ${PROJECT_IS_TOP_LEVEL})
if(IE_INSTALL)
    include(cmake/Install.cmake)
endif()

add_executable(inference main.cpp)
target_link_libraries(inference PRIVATE
    InferenceEngine::Generation
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
    InferenceEngine::Models
    InferenceEngine::UI
)

if(IE_ENABLE_CUDA)
    add_executable(llama_bench bench/llama_bench.cpp)
    target_link_libraries(llama_bench PRIVATE
        InferenceEngine::Generation
        InferenceEngine::NeuralNet
        InferenceEngine::Serialize
        InferenceEngine::Models
        InferenceEngine::CUDA
    )
endif()

include(CTest)
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
