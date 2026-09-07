#include "add_kernel.cuh"

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  out[tid] = in1[tid] + in2[tid];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, cudaStream_t stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  add_kernel_cu_fp32<<<block_num, thread_num, 0, stream>>>(
      size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
}
}  // namespace kernel
