#include "tokenizer/tokenizer.h"
#include <glog/logging.h>
#include <fstream>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "tokenizer/unicode.h"
namespace tokenizer {
// GPT-2 风格的分词正则,与 Qwen3 tokenizer 的训练模式一致。
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

// 词表 token 是 GPT-2 风格的字节展示形式(如空格显示为Ġ),编解码前做一次映射。
static void replace_all(std::string& s, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

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
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
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
  }

  stop_token1_ = special_tokens["<|im_end|>"];
  stop_token2_ = special_tokens["<|endoftext|>"];
  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(std::move(encoder), std::move(special_tokens),
                                                   PAT_STR);
}

std::vector<int32_t> Tokenizer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  std::string s = sentence;
  replace_all(s, " ", "Ġ");
  return this->tiktoken_->encode(s);
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  std::string s = tiktoken_->decode(token_ids);
  replace_all(s, "Ġ", " ");
  return s;
}

bool Tokenizer::is_sentence_ending(int32_t token_id) const {
  return token_id == stop_token1_ || token_id == stop_token2_;
}

int32_t Tokenizer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

}  // namespace tokenizer
