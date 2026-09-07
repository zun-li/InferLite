#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  init_buffer(alloc, data_type_, need_alloc, ptr);
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr) {
  if (need_alloc && alloc) {
    allocate(alloc);
  } else if (ptr) {
    // 指向外部内存(如 mmap 权重或张量切片)的视图。
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(base::DataTypeSize(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  }
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kUnknown;
  }
  return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (buffer_) {
    if (buffer_->device_type() != buffer->device_type()) {
      LOG(ERROR) << "The device type of the new buffer is different from the original one.";
    }
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

void Tensor::set_device_type(base::DeviceType device_type) const {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

// 调整形状。新形状不超过 buffer 容量时原地完成,否则扩容并搬数据。
void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  size_t byte_size = size * base::DataTypeSize(this->data_type_);
  if (byte_size > buffer_->byte_size()) {
    auto new_buffer = std::make_shared<base::Buffer>(byte_size, buffer_->allocator());
    CHECK(new_buffer->allocate());
    new_buffer->copy_from(buffer_.get());
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}
}  // namespace tensor
