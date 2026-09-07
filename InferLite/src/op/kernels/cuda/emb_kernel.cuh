#ifndef INFERLITE_KERNEL_EMB_CUH
#define INFERLITE_KERNEL_EMB_CUH
#include <cuda_runtime_api.h>
#include "tensor/tensor.h"
namespace kernel {
void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, cudaStream_t stream);
}
#endif  // INFERLITE_KERNEL_EMB_CUH
