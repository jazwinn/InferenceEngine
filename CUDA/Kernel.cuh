#pragma once
#include <cstddef>

#include "CUDA/DeviceBuffer.h"

namespace cuda{

    template <typename T>
    void LaunchMatMul(const DeviceBuffer<T>& dA, const DeviceBuffer<T>& dB, DeviceBuffer<T>& dC, size_t M, size_t N, size_t K);

    template <typename T>
    void LaunchMatMul(const T* A, const T* B, T* C, size_t M, size_t N, size_t K);

    
}