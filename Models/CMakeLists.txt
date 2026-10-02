find_package(nlohmann_json CONFIG REQUIRED)
find_package(pcre2 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(minja
    URL https://github.com/google/minja/archive/021c2293c187789ef13d56c6cfd89c9b134fd80f.tar.gz
    URL_HASH SHA256=dc3ddd37497b79a4cd35fd41550e22b3b0139439de8be6eab3d2d024fed47bb4
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR do-not-build
)
FetchContent_MakeAvailable(minja)

add_library(models INTERFACE)
add_library(InferenceEngine::Models ALIAS models)

target_link_libraries(models INTERFACE
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
    nlohmann_json::nlohmann_json
    PCRE2::8BIT
)

target_include_directories(models INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_include_directories(models SYSTEM INTERFACE
    $<BUILD_INTERFACE:${minja_SOURCE_DIR}/include>
)

file(GLOB_RECURSE MODELS_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(models INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES ${MODELS_HEADERS}
)
