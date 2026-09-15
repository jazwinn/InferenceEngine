find_package(GTest CONFIG REQUIRED)

add_executable(math_tests tensor_test.cpp)

target_link_libraries(math_tests PRIVATE
    InferenceEngine::Math
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(math_tests)
