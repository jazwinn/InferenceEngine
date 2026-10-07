#include "CUDA/DeviceBuffer.h"
#include "CUDA/detail/common.cuh"

namespace cuda{

    void* DeviceAlloc(size_t bytes){
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void DeviceFree(void* ptr){
        cudaFree(ptr);
    }

    void DeviceMemset(void* ptr, int val, size_t bytes){
        CUDA_CHECK(cudaMemset(ptr, val, bytes));
    }

    void CopyHostToDevice(void* dst, const void* src, size_t bytes){
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    }

    void CopyDeviceToHost(void* dst, const void* src, size_t bytes){
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }

    void CopyDeviceToDevice(void* dst, const void* src, size_t bytes){
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }

}
