#include "emb_kernel.cuh"
namespace kernel {
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, cudaStream_t stream) {
  CHECK(input.device_type() == base::DeviceType::kCUDA);
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kCUDA);

  constexpr int32_t thread_num = 128;
  // 每个 block 处理一个 token,block 数必须覆盖全部输入 token。
  const int32_t block_num = input_num > 0 ? input_num : 1;
  emb_kernel_cu_fp32<<<block_num, thread_num, 0, stream>>>(vocab_size, input_num, weight_dim,
                                                           input.ptr<int32_t>(),
                                                           weight.ptr<float>(),
                                                           const_cast<float*>(output.ptr<float>()));
}
}  // namespace kernel
