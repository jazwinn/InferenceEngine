find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(math_tests tensor_test.cpp)
target_link_libraries(math_tests PRIVATE
    InferenceEngine::Math
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(math_tests)

add_executable(nn_tests nn_test.cpp activation_test.cpp attentionops_test.cpp layernorm_test.cpp rmsnorm_test.cpp multiheadattention_test.cpp groupqueryattention_test.cpp kvcache_test.cpp kvcache_attention_test.cpp transformerconfig_test.cpp sequential_test.cpp embedding_test.cpp gpt2_test.cpp rope_test.cpp mlp_test.cpp decoder_test.cpp makenorm_test.cpp)
target_link_libraries(nn_tests PRIVATE
    InferenceEngine::NeuralNet
    InferenceEngine::Serialize
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(nn_tests)

add_executable(serialize_tests serialize_test.cpp safetensors_test.cpp)
target_link_libraries(serialize_tests PRIVATE
    InferenceEngine::Serialize
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(serialize_tests)

add_executable(generation_tests sampler_test.cpp temperature_test.cpp)
target_link_libraries(generation_tests PRIVATE
    InferenceEngine::Generation
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(generation_tests)
