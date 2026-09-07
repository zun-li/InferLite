#ifndef INFERLITE_INCLUDE_OP_EMBEDDING_H_
#define INFERLITE_INCLUDE_OP_EMBEDDING_H_
#include "layer.h"
namespace op {
// 查表式词嵌入:输入为 token id(主机内存),输出为对应的嵌入向量(显存)。
class EmbeddingLayer : public LayerParam {
 public:
  using Layer::forward;  // 避免子类的 forward() 隐藏基类多参重载

  EmbeddingLayer(int32_t dim, int32_t seq_len, int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
  tensor::Tensor device_tokens_;  // token id 的显存副本,forward 时从输入拷入
};
}  // namespace op
#endif  // INFERLITE_INCLUDE_OP_EMBEDDING_H_
