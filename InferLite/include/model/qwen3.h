#ifndef INFERLITE_INCLUDE_MODEL_QWEN3_H_
#define INFERLITE_INCLUDE_MODEL_QWEN3_H_
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "config.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/layer.h"
#include "op/mha.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "raw_model_data.h"
#include "tensor/tensor.h"
#include "tokenizer/tokenizer.h"
namespace model {
// 每层一份的算子集合,所有权重都在构造时上传到显存。
struct Qwen3Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::MultiHeadAttention> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::EmbeddingLayer> embedding_layer_;

  void set_stream(cudaStream_t stream);
};

// Qwen3-0.6B 的 CUDA 推理实现:加载 .bin 权重与 tokenizer.json,自回归生成。
class Qwen3Model {
 public:
  Qwen3Model(std::string token_path, std::string model_path);

  ~Qwen3Model();

  base::Status init();

  // 执行一次 forward(全部层),并采样出下一个 token。
  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const;

  // 把 token id 表查成嵌入向量(显存)。
  tensor::Tensor embedding(const std::vector<int>& tokens) const;

  std::vector<int32_t> encode(const std::string& sentence) const;

  std::string decode(std::vector<int32_t> token_idxs) const;

  bool is_sentence_ending(int32_t token_idx) const;

  tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  // 从嵌入向量中切出位置 pos 处的输入向量。
  tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                            const tensor::Tensor& input_embeddings, bool is_prompt) const;

 private:
  base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  base::Status read_model_file();

  base::Status generate_model_infos(const ModelConfig& config);

  base::Status gen_model_from_file();

  void init_mem();

  base::Status create_layers();

  void create_param_layers();

  void create_nonparam_layers();

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const;

  std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                           int32_t token_pos) const;

 private:
  std::unique_ptr<TransformerConfig> config_;
  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::shared_ptr<RawModelData> raw_model_data_;
  std::unique_ptr<Qwen3Layers> qwen_layers_;
  cudaStream_t stream_ = nullptr;
};
}  // namespace model
#endif  // INFERLITE_INCLUDE_MODEL_QWEN3_H_
