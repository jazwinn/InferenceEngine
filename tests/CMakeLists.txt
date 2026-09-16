find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(math_tests tensor_test.cpp)
target_link_libraries(math_tests PRIVATE
    InferenceEngine::Math
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(math_tests)

add_executable(nn_tests nn_test.cpp activation_test.cpp layernorm_test.cpp)
target_link_libraries(nn_tests PRIVATE
    InferenceEngine::NeuralNet
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(nn_tests)
