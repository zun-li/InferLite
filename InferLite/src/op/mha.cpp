#include "op/mha.h"
#include "kernels/cuda/mha_kernel.cuh"
namespace op {
MultiHeadAttention::MultiHeadAttention(int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  reset_input_size(4);
  reset_output_size(1);
}

base::Status MultiHeadAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  kernel::mha_kernel_cu(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_, head_size_,
                        this->get_output(0), this->get_input(0), this->get_input(1),
                        this->get_input(2), this->get_input(3), stream_);
  return base::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }

base::Status MultiHeadAttention::check() const {
  base::Status status;
  const int32_t input_tensor_num = 4;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    // mha score tensor
    status = check_tensor(get_input(i), base::DeviceType::kCUDA, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }
  return check_tensor(get_output(0), base::DeviceType::kCUDA, data_type_);
}

}  // namespace op
