#ifndef INFERLITE_KERNEL_ARGMAX_CUH
#define INFERLITE_KERNEL_ARGMAX_CUH
#include <cuda_runtime_api.h>
#include <cstddef>
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, cudaStream_t stream);
}
#endif  // INFERLITE_KERNEL_ARGMAX_CUH
