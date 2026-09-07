#include <glog/logging.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "model/qwen3.h"

// 把用户问题包装成 Qwen3 的对话模板。
std::string fill_template(const std::string& content) {
  const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos) {
    result.replace(pos, 2, content);
  }
  return result;
}

// 从 pos 处开始,逐 token 自回归生成,直到输出结束符或达到 total_steps。
// 生成 token 通过 need_output 决定是否解码打印。
int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      // 跳过模板符,只保留正文输出
      if (next != 151645 && next != 151644) {
        words.push_back(next);
      }
      if (model.is_sentence_ending(next)) {
        break;
      }
    }

    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }
  if (need_output) {
    printf("%s", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
  // 用法: ./infer <权重文件.bin> <tokenizer.json> [问题] [最大生成步数]
  if (argc < 3 || argc > 5) {
    LOG(INFO) << "Usage: ./infer <checkpoint> <tokenizer> [prompt] [max_steps]";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  std::string prompt = argc >= 4 ? argv[3] : "What is AI?";
  int total_steps = argc >= 5 ? std::atoi(argv[4]) : 2560;

  model::Qwen3Model model(tokenizer_path, checkpoint_path);
  auto init_status = model.init();
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: "
               << init_status.get_err_code();
  }

  std::cout << "Q: " << prompt << std::endl;
  std::cout << "A: ";
  fflush(stdout);
  const std::string& sentence = fill_template(prompt);
  auto start = std::chrono::steady_clock::now();
  int steps = generate(model, sentence, total_steps, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\n[steps:%d duration:%.3lfs speed:%.3lf tokens/s]\n", steps, duration,
         static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
