#ifndef INFERLITE_INCLUDE_TOKENIZER_UNICODE_H_
#define INFERLITE_INCLUDE_TOKENIZER_UNICODE_H_
#include <cstdint>
#include <string>
#include <vector>

namespace tokenizer {
// UTF-8 与字节级 BPE 展示形式之间的转换工具。
std::string unicode_cpt_to_utf8(uint32_t cp);

uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset);

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8);

uint8_t unicode_utf8_to_byte(const std::string& utf8);
}  // namespace tokenizer
#endif  // INFERLITE_INCLUDE_TOKENIZER_UNICODE_H_
