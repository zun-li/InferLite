#ifndef INFERLITE_INCLUDE_BASE_ALLOC_H_
#define INFERLITE_INCLUDE_BASE_ALLOC_H_
#include <memory>
#include "base.h"
namespace base {
// 统一的内存分配接口:Buffered 持有一个 allocator,析构时归还内存。
class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type);

  virtual ~DeviceAllocator() = default;

  DeviceType device_type() const;

  virtual void* allocate(size_t byte_size) const = 0;

  virtual void release(void* ptr) const = 0;

 private:
  DeviceType device_type_ = DeviceType::kUnknown;
};

// 主机内存:仅用于 token id 等小对象。
class HostAllocator : public DeviceAllocator {
 public:
  HostAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

// 显存:所有权重与中间张量都通过它分配。
class CudaAllocator : public DeviceAllocator {
 public:
  CudaAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

std::shared_ptr<DeviceAllocator> host_allocator();

std::shared_ptr<DeviceAllocator> cuda_allocator();
}  // namespace base
#endif  // INFERLITE_INCLUDE_BASE_ALLOC_H_
