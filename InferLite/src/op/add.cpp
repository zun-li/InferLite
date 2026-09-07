#include "op/add.h"
#include "kernels/cuda/add_kernel.cuh"
namespace op {
VecAddLayer::VecAddLayer() {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  int32_t size = input1.size();
  base::Status status;
  status = check_tensor_with_dim(input1, base::DeviceType::kCUDA, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(input2, base::DeviceType::kCUDA, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), base::DeviceType::kCUDA, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }
  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  kernel::add_kernel_cu(this->get_input(0), this->get_input(1), this->get_output(0), stream_);
  return base::error::Success();
}

}  // namespace op
