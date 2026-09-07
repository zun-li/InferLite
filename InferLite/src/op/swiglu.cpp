#include "op/swiglu.h"
#include "kernels/cuda/swiglu_kernel.cuh"
namespace op {
SwiGLULayer::SwiGLULayer(int32_t hidden_dim) : hidden_dim_(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status SwiGLULayer::check() const {
  base::Status status;
  const int32_t input_tensor_num = 2;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    status = check_tensor_with_dim(get_input(0), base::DeviceType::kCUDA, data_type_, hidden_dim_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the swiglu layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), base::DeviceType::kCUDA, data_type_, hidden_dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return status;
  }
  return base::error::Success();
}

base::Status SwiGLULayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  kernel::swiglu_kernel_cu(this->get_input(0), this->get_input(1), this->get_output(0), stream_);
  return base::error::Success();
}

}  // namespace op
