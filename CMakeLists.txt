cmake_minimum_required(VERSION 3.21)

# Must run before project(): the toolchain governs dependency resolution.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/vcpkg.cmake)

project(InferenceEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(Math)
add_subdirectory(NeuralNet)

include(CTest)
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
