#ifndef INFERLITE_KERNEL_SWIGLU_CUH
#define INFERLITE_KERNEL_SWIGLU_CUH
#include <cuda_runtime_api.h>
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, cudaStream_t stream);
}
#endif  // INFERLITE_KERNEL_SWIGLU_CUH
