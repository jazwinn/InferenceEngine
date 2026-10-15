#pragma once

#include "CUDA/detail/common.cuh"

namespace cuda
{
    template <typename T>
    struct AddOp{__device__ T operator()(T a, T b)const{return a + b;}};

    template <typename T>
    struct SubOp{__device__ T operator()(T a, T b)const{return a - b;}};

    template <typename T>
    struct MulOp{__device__ T operator()(T a, T b)const{return a * b;}};

    template <typename T>
    struct DivOp{__device__ T operator()(T a, T b)const{return a / b;}};

    template <typename T>
    struct SqrtOp{__device__ T operator()(T x)const{return sqrt(x);}};

    template <typename T>
    struct RELUops{
        __device__ T operator()(T x) const {
            return (x < T(0))? T(0) : x;
        }
    };

    template <typename T>
    struct GELUops{
        __device__ T operator()(T x) const {
            constexpr T ONE_OVER_SQRT_2 = T(0.7071067811865476);
            return T(0.5) * x * (T(1) + erf(x * ONE_OVER_SQRT_2));
        }
    };

    template <typename T>
    struct GELUTanhops{
        __device__ T operator()(T x) const {
            constexpr T SQRT_2_OVER_PI = T(0.7978845608028654);
            constexpr T COEFFICIENT = T(0.044715);
            const T inner = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
            return T(0.5) * x * (T(1) + tanh(inner));
        }
    };

    template <typename T>
    struct Sigmoidops{
        __device__ T operator()(T x) const {
            return T(1) / (T(1) + exp(-x));
        }
    };

    template <typename T>
    struct SiLUops{
        __device__ T operator()(T x) const {
            return x / (T(1) + exp(-x));
        }
    };

    template <typename T>
    struct Tanhops{
        __device__ T operator()(T x) const {
            return tanh(x);
        }
    };
} // namespace cuda
