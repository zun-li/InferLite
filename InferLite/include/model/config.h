#ifndef INFERLITE_INCLUDE_MODEL_CONFIG_H_
#define INFERLITE_INCLUDE_MODEL_CONFIG_H_
namespace model {
// 与导出脚本写入的 8 个 int32 文件头一一对应。
struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
  int32_t immediate_dim = 0;
};

// 运行时配置:由 ModelConfig 推导出的派生参数也放在这里。
struct TransformerConfig {
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t seq_len_ = 0;
  int32_t immediate_dim_ = 0;
};
}  // namespace model
#endif  // INFERLITE_INCLUDE_MODEL_CONFIG_H_
