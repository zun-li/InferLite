#include "base/alloc.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <glog/logging.h>
namespace base {
DeviceAllocator::DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

DeviceType DeviceAllocator::device_type() const { return device_type_; }

HostAllocator::HostAllocator() : DeviceAllocator(DeviceType::kCPU) {}

void* HostAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
  return std::malloc(byte_size);
}

void HostAllocator::release(void* ptr) const {
  if (ptr) {
    std::free(ptr);
  }
}

CudaAllocator::CudaAllocator() : DeviceAllocator(DeviceType::kCUDA) {}

void* CudaAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
  void* ptr = nullptr;
  cudaError_t state = cudaMalloc(&ptr, byte_size);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  return ptr;
}

void CudaAllocator::release(void* ptr) const {
  if (ptr) {
    CHECK(cudaFree(ptr) == cudaSuccess) << "Error: CUDA error when release memory on device";
  }
}

template <typename Allocator>
std::shared_ptr<DeviceAllocator> allocator_singleton() {
  static std::shared_ptr<DeviceAllocator> instance = std::make_shared<Allocator>();
  return instance;
}

std::shared_ptr<DeviceAllocator> host_allocator() {
  return allocator_singleton<HostAllocator>();
}

std::shared_ptr<DeviceAllocator> cuda_allocator() {
  return allocator_singleton<CudaAllocator>();
}

}  // namespace base
