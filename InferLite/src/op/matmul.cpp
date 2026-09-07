#include "op/matmul.h"
#include "kernels/cuda/matmul_kernel.cuh"
namespace op {
MatmulLayer::MatmulLayer(int32_t dim0, int32_t dim1) : dim0_(dim0), dim1_(dim1) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), base::DeviceType::kCUDA, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), base::DeviceType::kCUDA, data_type_, dim0_, dim1_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the matmul layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), base::DeviceType::kCUDA, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  kernel::matmul_kernel_cu(get_input(0), get_weight(0), get_output(0), stream_);
  return base::error::Success();
}

}  // namespace op
