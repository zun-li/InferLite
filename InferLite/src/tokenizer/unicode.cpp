#include "tokenizer/unicode.h"

#include <cassert>
#include <stdexcept>
#include <unordered_map>

namespace tokenizer {
namespace {
// 展示字符 -> 原始字节,镜像 GPT-2 字节级 BPE 约定:
// 可打印 ASCII (0x21-0x7E) 与 latin-1 的大部分字符映射为自身,
// 其余字节映射为码点 256 + n(n 为出现顺序)。
std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
  std::unordered_map<std::string, uint8_t> map;
  for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  auto n = 0;
  for (int ch = 0; ch < 256; ++ch) {
    if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
      map[unicode_cpt_to_utf8(256 + n)] = ch;
      ++n;
    }
  }
  return map;
}
}  // namespace

uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset) {
  assert(offset < utf8.size());
  if (!(utf8[offset + 0] & 0x80)) {
    auto result = utf8[offset + 0];
    offset += 1;
    return result;
  }
  if (!(utf8[offset + 0] & 0x40)) {
    throw std::invalid_argument("invalid character");
  }
  if (!(utf8[offset + 0] & 0x20)) {
    if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
    offset += 2;
    return result;
  }
  if (!(utf8[offset + 0] & 0x10)) {
    if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
        !((utf8[offset + 2] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto result =
        ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) |
        (utf8[offset + 2] & 0x3f);
    offset += 3;
    return result;
  }
  if (!(utf8[offset + 0] & 0x08)) {
    if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
        !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) |
                  ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
    offset += 4;
    return result;
  }
  throw std::invalid_argument("failed to convert utf8 to codepoint");
}

std::string unicode_cpt_to_utf8(uint32_t cp) {
  std::string result;

  if (cp <= 0x7f) {
    result.push_back(cp);
    return result;
  }
  if (0x80 <= cp && cp <= 0x7ff) {
    result.push_back(0xc0 | ((cp >> 6) & 0x1f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x800 <= cp && cp <= 0xffff) {
    result.push_back(0xe0 | ((cp >> 12) & 0x0f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x10000 <= cp && cp <= 0x10ffff) {
    result.push_back(0xf0 | ((cp >> 18) & 0x07));
    result.push_back(0x80 | ((cp >> 12) & 0x3f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }

  throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8) {
  std::vector<uint32_t> result;
  result.reserve(utf8.size());
  size_t offset = 0;
  while (offset < utf8.size()) {
    result.push_back(unicode_cpt_from_utf8(utf8, offset));
  }
  return result;
}

uint8_t unicode_utf8_to_byte(const std::string& utf8) {
  static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
  return map.at(utf8);
}
}  // namespace tokenizer
