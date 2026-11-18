# Install rules: `cmake --install <build> --prefix <dir>` produces
#
#   <dir>/include/...                         all public headers (+ minja)
#   <dir>/lib/cuda_kernels[d].lib             CUDA builds only
#   <dir>/lib/cmake/InferenceEngine/...       find_package(InferenceEngine)
#
# Consumers link the same targets as in-tree users: InferenceEngine::Tensor,
# ::NeuralNet, ::Models, ::Serialize, ::Reflection, ::Generation, ::UI (and
# ::CUDA when built with IE_ENABLE_CUDA).

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

set(_ie_exports
    tensor:Tensor
    neuralnet:NeuralNet
    reflection:Reflection
    serialize:Serialize
    models:Models
    generation:Generation
    ui:UI
)
if(IE_ENABLE_CUDA)
    list(APPEND _ie_exports cuda_kernels:CUDA)
endif()

set(_ie_targets "")
foreach(entry IN LISTS _ie_exports)
    string(REPLACE ":" ";" pair "${entry}")
    list(GET pair 0 target)
    list(GET pair 1 export_name)
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${export_name})
    list(APPEND _ie_targets ${target})
endforeach()

if(IE_ENABLE_CUDA)
    # Debug and Release libraries can live side by side in one package.
    set_target_properties(cuda_kernels PROPERTIES DEBUG_POSTFIX d)
endif()

install(TARGETS ${_ie_targets}
    EXPORT InferenceEngineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Models uses minja's chat-template headers, fetched at configure time.
FetchContent_GetProperties(minja)
if(minja_SOURCE_DIR)
    install(DIRECTORY ${minja_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

set(_ie_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/InferenceEngine)
install(EXPORT InferenceEngineTargets
    NAMESPACE InferenceEngine::
    DESTINATION ${_ie_cmake_dir}
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/InferenceEngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/InferenceEngineConfig.cmake
    INSTALL_DESTINATION ${_ie_cmake_dir}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/InferenceEngineConfigVersion.cmake
    VERSION ${IE_PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/InferenceEngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/InferenceEngineConfigVersion.cmake
    DESTINATION ${_ie_cmake_dir}
)
