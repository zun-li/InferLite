#ifndef INFERLITE_INCLUDE_BASE_BUFFER_H_
#define INFERLITE_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
// 一块内存的句柄:自持有(由 allocator 分配)或指向外部内存(如 mmap 权重、张量切片)。
class Buffer : public NoCopyable {
 private:
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::kUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);
};
}  // namespace base

#endif  // INFERLITE_INCLUDE_BASE_BUFFER_H_
