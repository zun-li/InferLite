#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>
namespace kernel {
// 一行权重对应一个 block:output[p] = sum_i(weight[p*M + i] * input[i])。
template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  const int row_offset = blockIdx.x * M;
  sdata[tid] = 0;
  float4* input_float4_ptr = (float4*)input;
  float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 input_float4 = *(input_float4_ptr + i);
    float4 weight_float4 = *(weight_float4_ptr + i);
    float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                     input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
    sdata[tid] += part_sum;
  }

  for (int i = pack_off + tid; i < M; i += blockDim.x) {
    sdata[tid] += input[i] * weight[row_offset + i];
  }

  __syncthreads();

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float part_sum = BlockReduce(temp).Sum(sdata[tid]);
  __syncthreads();

  if (tid == 0) {
    output[blockIdx.x] = part_sum;
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, cudaStream_t stream) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK_EQ(M, input.get_dim(0));
  matmul_kernel_cu_fp32<128><<<K, 128, 0, stream>>>(
      input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
}

}  // namespace kernel
