#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CUDA/Memory.h"
#include "CUDA/detail/common.cuh"

namespace cuda{

    namespace {

        constexpr size_t kMaxCached = 256ull << 20;
        constexpr size_t kSmallLimit = 1 << 20;
        constexpr size_t kSmallAlign = 512;
        constexpr size_t kLargeAlign = 2 << 20;

        size_t RoundUp(size_t bytes){
            const size_t align = bytes < kSmallLimit ? kSmallAlign : kLargeAlign;
            return CeilDiv(bytes, align) * align;
        }

        class CachingAllocator {
        public:
            CachingAllocator() = default;
            CachingAllocator(const CachingAllocator&) = delete;
            CachingAllocator& operator=(const CachingAllocator&) = delete;

            ~CachingAllocator(){
                for (auto& [size, blocks] : m_free) {
                    for (void* ptr : blocks) {
                        cudaFree(ptr);
                    }
                }
            }

            void* Allocate(size_t bytes){
                if (bytes == 0) return nullptr;

                std::lock_guard<std::mutex> lock(m_mutex);
                const size_t rounded = RoundUp(bytes);

                void* ptr = nullptr;

                auto it = m_free.find(rounded);
                if(it == m_free.end() || it->second.empty()){
                    cudaError_t status = cudaMalloc(&ptr, rounded);
                    if (status == cudaErrorMemoryAllocation) {
                        cudaGetLastError();
                        EmptyCacheLocked();
                        status = cudaMalloc(&ptr, rounded);
                    }
                    CUDA_CHECK(status);

                    m_allocated += rounded;
                    m_sizes[ptr] = rounded;
                }
                else{
                    ptr = it->second.back();
                    it->second.pop_back();
                    m_cached -= rounded;
                }

                m_live.insert(ptr);
                return ptr;
            }

            void Free(void* ptr){
                if(!ptr) return;

                std::lock_guard<std::mutex> lock(m_mutex);

                if (m_live.erase(ptr) == 0) {
                    throw std::invalid_argument("DeviceFree: pointer was not allocated by DeviceAlloc or was already freed.");
                }

                const size_t size = m_sizes.at(ptr);
                if (m_cached + size > kMaxCached) {
                    cudaFree(ptr);
                    m_sizes.erase(ptr);
                    m_allocated -= size;
                } else {
                    m_free[size].push_back(ptr);
                    m_cached += size;
                }
            }

            void EmptyCache(){
                std::lock_guard<std::mutex> lock(m_mutex);
                EmptyCacheLocked();
            }

            size_t CachedBytes() const {
                std::lock_guard<std::mutex> lock(m_mutex);
                return m_cached;
            }

            size_t AllocatedBytes() const {
                std::lock_guard<std::mutex> lock(m_mutex);
                return m_allocated;
            }

        private:
            void EmptyCacheLocked(){
                for (auto& [size, blocks] : m_free) {
                    for (void* ptr : blocks) {
                        cudaFree(ptr);
                        m_sizes.erase(ptr);
                        m_allocated -= size;
                    }
                }
                m_free.clear();
                m_cached = 0;
            }

            mutable std::mutex m_mutex;
            std::unordered_map<size_t, std::vector<void*>> m_free;
            std::unordered_map<void*, size_t> m_sizes;
            std::unordered_set<void*> m_live;
            size_t m_cached = 0;
            size_t m_allocated = 0;
        };

        CachingAllocator& Instance(){
            static CachingAllocator allocator;
            return allocator;
        }

    }

    void* DeviceAlloc(size_t bytes){
        return Instance().Allocate(bytes);
    }

    void DeviceFree(void* ptr){
        Instance().Free(ptr);
    }

    void EmptyCache(){
        Instance().EmptyCache();
    }

    size_t CachedBytes(){
        return Instance().CachedBytes();
    }

    size_t AllocatedBytes(){
        return Instance().AllocatedBytes();
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
