#pragma once

#include <re2/re2.h>

#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tiktoken {
// 字节级 BPE 分词(参考 openai/tiktoken 的 C++ 移植版)
// encode/decode 只依赖词表与正则,不依赖 torch/transformers

using Encoder = std::unordered_map<std::string, int>;

static auto _byte_pair_merge(const std::string& piece, const Encoder& ranks,
                             std::function<int(int, int)> func) -> std::vector<int> {
  std::vector<std::pair<int, int>> parts;
  parts.reserve(piece.size() + 1);
  for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
    parts.emplace_back(idx, std::numeric_limits<int>::max());
  }

  auto get_rank = [&piece, &ranks](const std::vector<std::pair<int, int>>& parts, int start_idx,
                                   int skip) -> std::optional<int> {
    if (start_idx + skip + 2 < parts.size()) {
      auto s = parts[start_idx].first;
      auto e = parts[start_idx + skip + 2].first;
      auto key = piece.substr(s, e - s);
      auto iter = ranks.find(key);
      if (iter != ranks.end()) {
        return iter->second;
      }
    }
    return std::nullopt;
  };

  for (auto i = 0U; i < parts.size() - 2; ++i) {
    auto rank = get_rank(parts, i, 0);
    if (rank) {
      assert(*rank != std::numeric_limits<int>::max());
      parts[i].second = *rank;
    }
  }

  while (true) {
    if (parts.size() == 1) break;

    auto min_rank = std::make_pair<int, int>(std::numeric_limits<int>::max(), 0);
    for (auto i = 0U; i < parts.size() - 1; ++i) {
      auto rank = parts[i].second;
      if (rank < min_rank.first) {
        min_rank = {rank, i};
      }
    }

    if (min_rank.first != std::numeric_limits<int>::max()) {
      auto i = min_rank.second;
      auto rank = get_rank(parts, i, 1);
      if (rank) {
        parts[i].second = *rank;
      } else {
        parts[i].second = std::numeric_limits<int>::max();
      }
      if (i > 0) {
        auto rank = get_rank(parts, i - 1, 1);
        if (rank) {
          parts[i - 1].second = *rank;
        } else {
          parts[i - 1].second = std::numeric_limits<int>::max();
        }
      }

      parts.erase(parts.begin() + (i + 1));
    } else {
      break;
    }
  }
  std::vector<int> out;
  out.reserve(parts.size() - 1);
  for (auto i = 0U; i < parts.size() - 1; ++i) {
    out.push_back(func(parts[i].first, parts[i + 1].first));
  }
  return out;
}

static auto byte_pair_encode(const std::string& piece, const Encoder& ranks) -> std::vector<int> {
  if (piece.size() == 1) {
    return {ranks.at(piece)};
  }

  auto func = [&piece, &ranks](int start, int stop) -> int {
    std::string key = piece.substr(start, stop - start);
    return ranks.at(key);
  };

  return _byte_pair_merge(piece, ranks, func);
}

class tiktoken {
 public:
  tiktoken() = default;

  tiktoken(Encoder encoder, Encoder special_encoder, const std::string& pattern) {
    regex_ = std::make_unique<re2::RE2>("(" + pattern + ")");
    if (!regex_->ok()) {
      throw std::runtime_error("Invalid tokenizer pattern: " + regex_->error());
    }

    std::string special_pattern;
    for (const auto& item : special_encoder) {
      if (!special_pattern.empty()) {
        special_pattern += "|";
      }
      special_pattern += re2::RE2::QuoteMeta(item.first);
    }
    if (special_pattern.empty()) {
      special_regex_ = nullptr;
    } else {
      special_regex_ = std::make_unique<re2::RE2>("(" + special_pattern + ")");
      if (!special_regex_->ok()) {
        throw std::runtime_error("Invalid special token pattern: " + special_regex_->error());
      }
    }

    encoder_ = std::move(encoder);
    special_tokens_encoder = std::move(special_encoder);

    for (const auto& [k, v] : encoder_) {
      decoder_.emplace(v, k);
    }

    for (const auto& [k, v] : special_tokens_encoder) {
      special_tokens_decoder.emplace(v, k);
    }
  }

  auto encode(const std::string& text) const -> std::vector<int> {
    return _encode_native(text, special_tokens_encoder).first;
  }

  auto decode(const std::vector<int>& tokens) const -> std::string { return _decode_native(tokens); }

 private:
  auto split_with_allowed_special_token(
      re2::StringPiece& input,
      const Encoder& allowed_special) const -> std::pair<std::optional<std::string>, re2::StringPiece> {
    if (special_regex_ == nullptr) return {std::nullopt, input};

    auto start = input.begin();
    std::string special;
    while (true) {
      if (!re2::RE2::FindAndConsume(&input, *special_regex_, &special)) {
        break;
      }

      if (allowed_special.count(special) == 1) {
        return {std::move(special),
                re2::StringPiece(start, input.begin() - start - special.size())};
      }
    }

    return {std::nullopt, input};
  }

  // RE2 的 \s 等价于 [ \t\n\f\r]。
  static auto _is_re2_space(char c) -> bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\f' || c == '\r';
  }

  // 只由空格/制表/换页组成且长度 >= 2 的分片:即不含换行的连续空白,
  // 这种分片只可能由 PAT_STR 的 `\s+(?:$|[^\S])` 产生。
  static auto _is_non_newline_space_run(const std::string& piece) -> bool {
    if (piece.size() < 2) return false;
    for (const char c : piece) {
      if (c != ' ' && c != '\t' && c != '\f') return false;
    }
    return true;
  }

  auto _encode_native(const std::string& text,
                      const Encoder& allowed_special) const -> std::pair<std::vector<int>, int> {
    std::vector<int> ret;
    int last_piece_token_len = 0;
    re2::StringPiece input(text);

    while (true) {
      auto [special, sub_input] = split_with_allowed_special_token(input, allowed_special);
      re2::StringPiece rest = sub_input;
      while (true) {
        re2::StringPiece before = rest;
        std::string piece;
        if (!re2::RE2::FindAndConsume(&rest, *regex_, &piece)) {
          break;
        }

        // PAT_STR 用 `\s+(?:$|[^\S])` 代替官方的 `\s+(?!\S)`(RE2 不支持前瞻断言),
        // 前者会把一段不含换行的连续空白整段吃掉,官方只吃 L-1 个并把最后一个空白
        // 留给 `[^\r\n\p{L}\p{N}]?\p{L}+` / ` ?[^\s\p{L}\p{N}]+`。这里退回最后一个
        // 空白,让它与后面的文本一起重新匹配,分片结果即与官方 pattern 一致。
        if (_is_non_newline_space_run(piece) && rest.begin() < before.end() &&
            !_is_re2_space(*rest.begin())) {
          piece.pop_back();
          rest = re2::StringPiece(rest.begin() - 1,
                                  static_cast<size_t>(before.end() - (rest.begin() - 1)));
        }

        auto iter = encoder_.find(piece);
        if (iter != encoder_.end()) {
          last_piece_token_len = 1;
          ret.push_back(iter->second);
          continue;
        }
        auto tokens = byte_pair_encode(piece, encoder_);
        last_piece_token_len = tokens.size();
        ret.insert(ret.end(), tokens.begin(), tokens.end());
      }

      if (special) {
        int token = special_tokens_encoder.at(*special);
        ret.push_back(token);
        last_piece_token_len = 0;
      } else {
        break;
      }
    }

    return {ret, last_piece_token_len};
  }

  auto _decode_native(const std::vector<int>& tokens) const -> std::string {
    std::string ret;
    ret.reserve(tokens.size() * 2);
    for (auto token : tokens) {
      std::string token_bytes;
      auto iter = decoder_.find(token);
      if (iter != decoder_.end()) {
        token_bytes = iter->second;
      } else {
        iter = special_tokens_decoder.find(token);
        if (iter != special_tokens_decoder.end()) {
          token_bytes = iter->second;
        } else {
          throw std::runtime_error("unknown token: " + std::to_string(token));
        }
      }
      ret += token_bytes;
    }
    return ret;
  }

  Encoder encoder_;
  Encoder special_tokens_encoder;
  std::unordered_map<int, std::string> decoder_;
  std::unordered_map<int, std::string> special_tokens_decoder;
  std::unique_ptr<re2::RE2> regex_;
  std::unique_ptr<re2::RE2> special_regex_;
};

}  // namespace tiktoken
