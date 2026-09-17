#include "tokenizer/tokenizer.h"
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "tokenizer/unicode.h"
namespace tokenizer {
// GPT-2 风格的分词正则,与 Qwen3 tokenizer 的训练模式一致。
// 官方 pattern 的 `\s+(?!\S)` 含前瞻断言,RE2 不支持,改用 `\s+(?:$|[^\S])`;
// 两者的差异由 tiktoken::_encode_native 里的空白分片修正补齐。
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

Tokenizer::Tokenizer(std::string token_model_path) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path);
  CHECK(f.is_open())
      << "The token model path is not valid, please check the path and type of token model.";
  json data;
  try {
    data = json::parse(f);
  } catch (json::parse_error&) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }

  const auto& datas = data["added_tokens"];
  std::unordered_map<std::string, int> special_tokens;
  int32_t max_token_id = -1;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
    max_token_id = std::max(max_token_id, id);
  }
  CHECK(special_tokens.count("<|im_start|>") == 1 && special_tokens.count("<|im_end|>") == 1 &&
        special_tokens.count("<|endoftext|>") == 1)
      << "The token model is not a Qwen3 tokenizer, the special tokens are missing.";

  // 词表中的 token 是 GPT-2 风格的字节展示形式(如空格显示为Ġ),
  // 逐字符转回字节串后作为 BPE 词表键。
  tiktoken::Encoder encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
    max_token_id = std::max(max_token_id, id);
  }

  stop_token1_ = special_tokens["<|im_end|>"];
  stop_token2_ = special_tokens["<|endoftext|>"];
  // 词表大小取最大 token id + 1,而不是两份词表大小相加:added_tokens 与
  // model.vocab 的 id 可能重叠,相加会重复计数。
  num_token_ = max_token_id + 1;
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(std::move(encoder), std::move(special_tokens),
                                                   PAT_STR);
}

std::vector<int32_t> Tokenizer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  // 词表键在构造函数里已经从 GPT-2 展示形式转成原始字节(空格就是 0x20),
  // 这里直接编码原文即可,不能再把空格替换成 'Ġ'。
  return this->tiktoken_->encode(sentence);
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  // decoder_ 存的同样是原始字节,直接拼接即可,不需要把 'Ġ' 换回空格。
  return this->tiktoken_->decode(token_ids);
}

bool Tokenizer::is_sentence_ending(int32_t token_id) const {
  return token_id == stop_token1_ || token_id == stop_token2_;
}

int32_t Tokenizer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

}  // namespace tokenizer
