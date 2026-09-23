cmake_minimum_required(VERSION 3.21)

# Must run before project(): the toolchain governs dependency resolution.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/vcpkg.cmake)

project(InferenceEngine LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single-config generators" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(Generation)
add_subdirectory(Math)
add_subdirectory(NeuralNet)
add_subdirectory(Reflection)
add_subdirectory(Serialize)

add_executable(inference main.cpp)
target_link_libraries(inference PRIVATE
    InferenceEngine::Generation
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
)

include(CTest)
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
