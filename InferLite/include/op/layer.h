#ifndef INFERLITE_INCLUDE_OP_LAYER_H_
#define INFERLITE_INCLUDE_OP_LAYER_H_
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
// 算子的统一基类。所有算子都运行在 CUDA 上。
class Layer {
 public:
  explicit Layer() = default;

  virtual ~Layer() = default;

  // 算子内部的数据/状态检查,forward 前调用。
  virtual base::Status check() const = 0;

  // 真实计算入口:子类通过 get_input/get_output 取数据。
  virtual base::Status forward() = 0;

  // 多参 forward 仅做输入/输出绑定,再转发到零参 forward()。
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1);

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1);

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1);

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1);

  void set_input(int32_t idx, const tensor::Tensor& input);

  void set_output(int32_t idx, const tensor::Tensor& output);

  tensor::Tensor& get_input(int32_t idx);

  const tensor::Tensor& get_input(int32_t idx) const;

  tensor::Tensor& get_output(int32_t idx);

  const tensor::Tensor& get_output(int32_t idx) const;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;

  base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  void set_stream(cudaStream_t stream);

 protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  cudaStream_t stream_ = nullptr;
  base::DataType data_type_ = base::DataType::kFp32;
};

// 带权重的算子:set_weight 把主机上的权重直接上传到显存。
class LayerParam : public Layer {
 public:
  explicit LayerParam() = default;

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void reset_weight_size(size_t size);

  base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr);

 protected:
  std::vector<tensor::Tensor> weights_;
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_LAYER_H_
