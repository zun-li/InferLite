#include "model/qwen3.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <utility>
#include "../op/kernels/cuda/argmax_kernel.cuh"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "op/matmul.h"
#include "op/rmsnorm.h"
namespace model {

void Qwen3Layers::set_stream(cudaStream_t stream) {
  for (auto* layer : std::vector<op::Layer*>{add_layer_.get(), rope_layer_.get(),
                                             swiglu_layer_.get(), mha_layer_.get(),
                                             cls_layer_.get(), embedding_layer_.get()}) {
    if (layer) {
      layer->set_stream(stream);
    }
  }
  for (auto* group :
       std::vector<std::vector<std::shared_ptr<op::Layer>>*>{&wq_layers_, &wk_layers_, &wv_layers_,
                                                             &wo_layers_, &w1_layers_, &w2_layers_,
                                                             &w3_layers_, &rmsnorm_layers_}) {
    for (auto& layer : *group) {
      layer->set_stream(stream);
    }
  }
}

Qwen3Model::Qwen3Model(std::string token_path, std::string model_path)
    : token_path_(std::move(token_path)), model_path_(std::move(model_path)) {
  tokenizer_ = std::make_unique<tokenizer::Tokenizer>(token_path_);
}

Qwen3Model::~Qwen3Model() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

base::Status Qwen3Model::init() {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  cudaSetDevice(0);
  cudaStreamCreate(&stream_);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return error::InternalError("Failed to create the CUDA stream.");
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                get_buffer(ModelBufferType::kSinCache),
                                get_buffer(ModelBufferType::kCosCache), stream_);
  qwen_layers_->set_stream(stream_);
  return error::Success();
}

base::Status Qwen3Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

base::Status Qwen3Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    fclose(file);
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model file.");
  }
  fclose(file);

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    close(fd);
    return gen_status;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model file.");
  }

  raw_model_data_ = std::make_shared<RawModelData>();
  raw_model_data_->file_size = sb.st_size;
  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
  if (raw_model_data_->data == MAP_FAILED) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory.");
  }
  raw_model_data_->weight_data =
      static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  return error::Success();
}

base::Status Qwen3Model::generate_model_infos(const ModelConfig& config) {
  // Qwen3-0.6B 的关键结构参数,用于拒绝其它架构/尺寸的权重文件。
  constexpr int32_t kDim = 2048;
  constexpr int32_t kHiddenDim = 1024;
  constexpr int32_t kLayerNum = 28;
  constexpr int32_t kHeadNum = 16;
  constexpr int32_t kKvHeadNum = 8;
  constexpr int32_t kVocabSize = 151936;
  constexpr int32_t kIntermediateDim = 3072;
  if (config.dim != kDim || config.hidden_dim != kHiddenDim ||
      config.layer_num != kLayerNum || config.head_num != kHeadNum ||
      config.kv_head_num != kKvHeadNum || std::abs(config.vocab_size) != kVocabSize ||
      config.immediate_dim != kIntermediateDim || config.seq_len <= 0) {
    return base::error::ModelParseError("The checkpoint architecture is not Qwen3-0.6B.");
  }
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;
  config_->immediate_dim_ = config.immediate_dim;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Qwen3Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Qwen3Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Qwen3Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

std::vector<int32_t> Qwen3Model::encode(const std::string& sentence) const {
  CHECK(tokenizer_ != nullptr);
  return tokenizer_->encode(sentence);
}

bool Qwen3Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(tokenizer_ != nullptr);
  return tokenizer_->is_sentence_ending(token_idx);
}

std::string Qwen3Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(tokenizer_ != nullptr);
  return tokenizer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Qwen3Model::slice_kv_cache(int32_t layer_idx,
                                                                     int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kFp32, config_->kv_dim_, false, nullptr, key_cache_ptr);
  tensor::Tensor val(base::DataType::kFp32, config_->kv_dim_, false, nullptr, val_cache_ptr);
  key.set_device_type(base::DeviceType::kCUDA);
  val.set_device_type(base::DeviceType::kCUDA);
  return {key, val};
}

tensor::Tensor Qwen3Model::fill_input(const tensor::Tensor& pos_tensor,
                                      const tensor::Tensor& input_embeddings,
                                      bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      const_cast<float*>(input_embeddings.ptr<float>(index * config_->hidden_dim_)), true);
  tensor::Tensor input(base::DataType::kFp32, config_->hidden_dim_);
  input.assign(input_emb_buffer);
  input.set_device_type(base::DeviceType::kCUDA);
  return input;
}

base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void Qwen3Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ =
      std::make_shared<op::RoPELayer>(config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>();

  qwen_layers_->swiglu_layer_ = std::make_shared<op::SwiGLULayer>(config_->immediate_dim_);
}

void Qwen3Model::create_param_layers() {
  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  float* weight_ptr = (float*)raw_model_data_->weight(pos);

  // rmsnorm: attention 输入 L 个 + attention 输出 L 个 + 最终 1 个
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(hidden_dim);
    rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += hidden_dim;
  }
  pos += (2 * config_->layer_num_ + 1) * hidden_dim;

  // embedding layer
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      hidden_dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(
      0, {std::abs(config_->vocab_size_), hidden_dim}, weight_ptr);
  pos += config_->vocab_size_ * hidden_dim;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(dim, hidden_dim);
    wq->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + hidden_dim * dim;
  }

  // query norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos));
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(hidden_dim, kv_dim);
    wk->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + hidden_dim * kv_dim;
  }

  // key norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos));
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(hidden_dim, kv_dim);
    wv->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->wv_layers_.push_back(wv);
    pos += kv_dim * hidden_dim;
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(hidden_dim, dim);
    wo->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * hidden_dim;
  }

  // w1 layers (gate)
  int32_t immediate_dim = config_->immediate_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(immediate_dim, hidden_dim);
    w1->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + hidden_dim * immediate_dim;
  }

  // w2 layers (down)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(hidden_dim, immediate_dim);
    w2->set_weight(0, {hidden_dim, immediate_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + immediate_dim * hidden_dim;
  }

  // w3 layers (up)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(immediate_dim, hidden_dim);
    w3->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos));
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + immediate_dim * hidden_dim;
  }

  auto lm_head =
      std::make_shared<op::MatmulLayer>(config_->vocab_size_, config_->hidden_dim_);
  lm_head->set_weight(0, {config_->vocab_size_, config_->hidden_dim_},
                      this->raw_model_data_->weight(pos));
  qwen_layers_->cls_layer_ = lm_head;
}

void Qwen3Model::init_mem() {
  auto alloc_cu = base::cuda_allocator();
  auto alloc_host = base::host_allocator();

  // 输入按 seq_len 容量预分配,后续 embedding 只需原地 reshape。
  tensor::Tensor input_tokens(base::DataType::kInt32, config_->seq_len_, true, alloc_host);
  tensor::Tensor input_embeddings(base::DataType::kFp32, config_->seq_len_, config_->hidden_dim_,
                                  true, alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor sin_cache(base::DataType::kFp32, config_->head_size_ * config_->seq_len_, true,
                           alloc_cu);
  tensor::Tensor cos_cache(base::DataType::kFp32, config_->head_size_ * config_->seq_len_, true,
                           alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  tensor::Tensor rms_output(base::DataType::kFp32, config_->hidden_dim_, true, alloc_cu);
  tensor::Tensor out_mha(base::DataType::kFp32, config_->dim_, true, alloc_cu);

  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kFp32, config_->immediate_dim_, true, alloc_cu);
  tensor::Tensor w3_output(base::DataType::kFp32, config_->immediate_dim_, true, alloc_cu);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc_cu);
  tensor::Tensor value_cache(base::DataType::kFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc_cu);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kFp32, config_->dim_, true, alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kInt32, 1, true, alloc_host);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention score/output
  tensor::Tensor attn(base::DataType::kFp32, config_->head_num_, config_->seq_len_, true,
                      alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  tensor::Tensor attn_output(base::DataType::kFp32, config_->hidden_dim_, true, alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kFp32, config_->vocab_size_, true, alloc_cu);
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status Qwen3Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  create_param_layers();
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for Qwen3 failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
    // input norm
    return error::InternalError("Create the rmsnorm layers for Qwen3 failed!");
  }

  for (auto* group : std::vector<std::vector<std::shared_ptr<op::Layer>>*>{
           &qwen_layers_->wq_layers_, &qwen_layers_->wk_layers_, &qwen_layers_->wv_layers_,
           &qwen_layers_->wo_layers_, &qwen_layers_->w1_layers_, &qwen_layers_->w2_layers_,
           &qwen_layers_->w3_layers_}) {
    if (group->size() != config_->layer_num_) {
      return error::InternalError("Create the matmul layers for Qwen3 failed!");
    }
  }

  for (auto& qwen_layer : qwen_layers_->wq_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the query layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->wk_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the key layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->wv_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the value layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->wo_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the output layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->w1_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the w1 layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->w2_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the w2 layer for Qwen3 failed!");
    }
  }
  for (auto& qwen_layer : qwen_layers_->w3_layers_) {
    if (!qwen_layer) {
      return error::InternalError("Create the w3 layer for Qwen3 failed!");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for Qwen3 failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for Qwen3 failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the MHA layer for Qwen3 failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for Qwen3 failed!");
  }
  return error::Success();
}

void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK_NE(rmsnorm_layer, nullptr) << "The attention RMSNorm layer is null in the Qwen3 model";
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // query norm
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // key norm
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(key, key);
  key.reshape({(int32_t)key.size()});

  // value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(qwen_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(qwen_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = qwen_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  mha_layer->set_pos(pos);
  mha_layer->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm (post attention layernorm)
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

tensor::Tensor Qwen3Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  input_tokens.reshape({static_cast<int32_t>(tokens.size())});
  input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->hidden_dim_});
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the Qwen3 model is null.";
  STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_embeddings));
  return input_embeddings;
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen3Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  if (is_prompt) {
    return -1;
  }
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  return static_cast<int32_t>(
      kernel::argmax_kernel_cu(forward_output.ptr<float>(), forward_output.size(), stream_));
}

}  // namespace model
