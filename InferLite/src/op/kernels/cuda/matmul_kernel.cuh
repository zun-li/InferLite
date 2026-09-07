#ifndef INFERLITE_KERNEL_MATMUL_CUH
#define INFERLITE_KERNEL_MATMUL_CUH
#include <cuda_runtime_api.h>
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, cudaStream_t stream);
}  // namespace kernel

#endif  // INFERLITE_KERNEL_MATMUL_CUH
