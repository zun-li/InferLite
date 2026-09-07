#ifndef INFERLITE_KERNEL_RMSNORM_CUH
#define INFERLITE_KERNEL_RMSNORM_CUH
#include <cuda_runtime_api.h>
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, cudaStream_t stream);

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, cudaStream_t stream);
}  // namespace kernel
#endif  // INFERLITE_KERNEL_RMSNORM_CUH
