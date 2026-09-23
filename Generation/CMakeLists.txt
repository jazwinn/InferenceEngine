add_library(generation INTERFACE)
add_library(InferenceEngine::Generation ALIAS generation)

target_link_libraries(generation INTERFACE InferenceEngine::Math)

target_include_directories(generation INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

file(GLOB_RECURSE GENERATION_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(generation INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ${GENERATION_HEADERS}
)
