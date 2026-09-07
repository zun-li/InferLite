#include "op/rope.h"
#include <cmath>
#include "kernels/cuda/rope_kernel.cuh"
namespace op {
RoPELayer::RoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size)
    : dim_(dim), kv_dim_(kv_dim), head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status RoPELayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  kernel::rope_kernel_cu(dim_, kv_dim_, head_size_, this->get_input(0), this->get_input(1),
                         this->get_input(2), this->get_input(3), this->get_input(4), stream_);
  return base::error::Success();
}

base::Status RoPELayer::check() const {
  // pos tensor(主机内存上的 int32)
  auto status = check_tensor_with_dim(get_input(2), base::DeviceType::kCPU,
                                      base::DataType::kInt32, 1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the rope layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(1), base::DeviceType::kCUDA, data_type_, kv_dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the rope layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(0), base::DeviceType::kCUDA, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the rope layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op
