cmake_minimum_required(VERSION 3.24)

# Must run before project(): the toolchain governs dependency resolution.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/vcpkg.cmake)

option(IE_ENABLE_CUDA "Build the CUDA backend" ON)

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
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive-> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
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

add_executable(inference main.cpp)
target_link_libraries(inference PRIVATE
    InferenceEngine::Generation
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
    InferenceEngine::Models
)

include(CTest)
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
