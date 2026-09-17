#ifndef INFERLITE_INCLUDE_TOKENIZER_TOKENIZER_H_
#define INFERLITE_INCLUDE_TOKENIZER_TOKENIZER_H_
#include <memory>
#include <string>
#include <vector>
#include "tiktoken.h"
namespace tokenizer {
// Qwen3 的字节级 BPE 分词:解析 HF tokenizer.json,提供 encode / decode,
// 并识别 <|im_end|> / <|endoftext|> 结束符。分词始终在主机上执行。
class Tokenizer {
 public:
  explicit Tokenizer(std::string token_model_path);

  std::vector<int32_t> encode(const std::string& sentence) const;

  std::string decode(const std::vector<int32_t>& token_ids) const;

  bool is_sentence_ending(int32_t token_id) const;

  // 词表最大 token id + 1(本模型为 151669)。注意它不是模型 embedding/lm_head
  // 的行数:权重里的 vocab_size 是补齐到 151936 的值,分配 logits 时要用后者。
  int32_t vocab_size() const;

 private:
  int32_t stop_token1_ = -1;
  int32_t stop_token2_ = -1;
  int32_t num_token_ = 0;
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};
}  // namespace tokenizer
#endif  // INFERLITE_INCLUDE_TOKENIZER_TOKENIZER_H_
