#include "op/embedding.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/emb_kernel.cuh"
namespace op {
EmbeddingLayer::EmbeddingLayer(int32_t dim, int32_t seq_len, int32_t vocab_size)
    : dim_(dim), seq_len_(seq_len), vocab_size_(vocab_size) {
  reset_weight_size(1);
  reset_input_size(1);
  reset_output_size(1);
  device_tokens_ = tensor::Tensor(base::DataType::kInt32, seq_len_);
  CHECK(device_tokens_.allocate(base::cuda_allocator()));
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  base::Status status = check_tensor(input_tensor, base::DeviceType::kCPU,
                                     base::DataType::kInt32);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), base::DeviceType::kCUDA, data_type_, vocab_size_,
                                 dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), base::DeviceType::kCUDA, data_type_,
                                 input_tensor.size(), dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  const auto& input = get_input(0);
  const auto& output = get_output(0);
  // token id 在主机上填充,这里拷入显存后交给 kernel 查表。
  cudaMemcpyAsync(device_tokens_.ptr<int32_t>(), input.ptr<int32_t>(), input.byte_size(),
                  cudaMemcpyHostToDevice, stream_);
  kernel::emb_kernel_cu(device_tokens_, get_weight(0), output, vocab_size_, stream_);
  return base::error::Success();
}
}  // namespace op
