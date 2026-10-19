#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace cuda{

    void* DeviceAlloc(size_t bytes);
    void DeviceFree(void* ptr);
    void EmptyCache();
    size_t CachedBytes();
    size_t AllocatedBytes();
    void CopyHostToDevice(void* dst, const void* src, size_t bytes);
    void CopyDeviceToHost(void* dst, const void* src, size_t bytes);
    void CopyDeviceToDevice(void* dst, const void* src, size_t bytes);
    void DeviceMemset(void* ptr, int val, size_t bytes);

    template <typename T>
    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(size_t count){
            Malloc(count);
        }

        explicit DeviceBuffer(const T* src, size_t count):
            DeviceBuffer(count)
        {
            CopyFromHost(src, count);
        }

        ~DeviceBuffer(){
            Free();
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept :
            m_buffer(std::exchange(other.m_buffer, nullptr)),
            m_count(std::exchange(other.m_count, 0))
        {}

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept{
            if (this != &other) {
                Free();
                m_buffer = std::exchange(other.m_buffer, nullptr);
                m_count = std::exchange(other.m_count, 0);
            }
            return *this;
        }

        void Resize(size_t count){
            if(count == m_count) return;
            Malloc(count);
        }

        void CopyFromHost(const T* src, size_t count){
            if(count > m_count){
                throw std::invalid_argument("CopyFromHost: count exceeds buffer size");
            }
            CopyHostToDevice(m_buffer, src, count * sizeof(T));
        }

        void CopyToHost(T* dst, size_t count) const{
            if(count > m_count){
                throw std::invalid_argument("CopyToHost: count exceeds buffer size");
            }
            CopyDeviceToHost(dst, m_buffer, count * sizeof(T));
        }

        void CopyToDevice(T* dst, size_t count) const{
            if(count > m_count){
                throw std::invalid_argument("CopyToDevice: count exceeds buffer size");
            }
            CopyDeviceToDevice(dst, m_buffer, count * sizeof(T));
        }

        void Memset(int val, size_t count){
            if(count > m_count){
                throw std::invalid_argument("Memset: count exceeds buffer size");
            }
            if(count > 0){
                DeviceMemset(m_buffer, val, count * sizeof(T));
            }
        }

        T* Data(){return m_buffer;}
        const T* Data() const{return m_buffer;}
        size_t Size() const{return m_count;}
        size_t Bytes() const{return m_count * sizeof(T);}

    private:

        void Malloc(size_t count){
            Free();
            if(count > 0){
                m_buffer = static_cast<T*>(DeviceAlloc(count * sizeof(T)));
            }
            m_count = count;
        }

        void Free(){
            DeviceFree(m_buffer);
            m_buffer = nullptr;
            m_count = 0;
        }

        T* m_buffer = nullptr;
        size_t m_count = 0;
    };

}
