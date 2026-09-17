// 推理 demo：加载 Qwen3-0.6B 权重，对一条 prompt 做自回归生成并打印结果
#include <glog/logging.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "model/qwen3.h"

// 把用户问题包装成 Qwen3 的对话模板
std::string fill_template(const std::string &prompt) {
    const std::string format =
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";

    std::string formatted_prompt = format;

    size_t pos = formatted_prompt.find("%s");
    if (pos != std::string::npos) {
        formatted_prompt.replace(pos, 2, prompt);
    }

    return formatted_prompt;
}

// 逐 token 自回归生成，返回实际执行的步数(含 prompt 阶段)
int32_t generate(const model::Qwen3Model &model, const std::string &formatted_prompt,
                 int max_steps, bool print_output = false) {

    // 文本 → token id
    const std::vector<int32_t> prompt_tokens = model.encode(formatted_prompt);
    const int32_t prompt_len = prompt_tokens.size();
    LOG_IF(FATAL, prompt_tokens.empty()) << "The tokens is empty.";

    // 当前处理的序列位置
    int32_t pos = 0;
    // 下一步要喂入的 token
    int32_t next_token = prompt_tokens.at(pos);
    // 是否处于 prefill 阶段
    bool is_prefill = true;
    // 整段 prompt 的嵌入矩阵
    const auto &prompt_embeddings = model.embedding(prompt_tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    // 收集待输出的正文 token
    std::vector<int32_t> output_tokens;

    // 自回归生成
    while (pos < max_steps) {

        // 位置驱动 RoPE、KV cache 槽位与因果掩码
        pos_tensor.index<int32_t>(0) = pos;

        // prefill：逐 token 建立 KV cache，此阶段不做采样
        if (pos < prompt_len - 1) {
            tensor::Tensor step_input = model.fill_input(pos_tensor, prompt_embeddings, is_prefill);
            model.predict(step_input, pos_tensor, is_prefill, next_token);
        }

        // decode：只嵌入一个 token，由 argmax 采样出下一个 token
        else {
            
            // 首次进入 decode 时，输入仍是最后一个 prompt token
            if (pos == prompt_len - 1) {
                next_token = prompt_tokens.back();
            }
            
            is_prefill = false;
            std::vector<int32_t> step_tokens{next_token};
            const auto &step_embedding = model.embedding(step_tokens);
            tensor::Tensor step_input = model.fill_input(pos_tensor, step_embedding, is_prefill);
            model.predict(step_input, pos_tensor, is_prefill, next_token);

            // 跳过模板符，只保留正文输出
            if (next_token != 151645 && next_token != 151644) {
                output_tokens.push_back(next_token);
            }

            // 命中 <|im_end|> 或者 <|endoftext|> 即停止
            if (model.is_sentence_ending(next_token)) {
                break;
            }
        }

        pos += 1;
    }

    // 如果函数传入的参数 print_output 为 true，就打印结果
    if (print_output) {
        printf("%s", model.decode(output_tokens).data());
        fflush(stdout);
    }

    // 实际执行的步数(含 prompt 阶段)
    return std::min(pos, max_steps);
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 5)
    {
        LOG(INFO) << "Usage: ./infer <checkpoint> <tokenizer> [prompt] [max_steps]";
        return -1;
    }
    
    // 模型权重和分词的路径
    const char *checkpoint_path = argv[1];
    const char *tokenizer_path = argv[2];
    
    // 用户输入的提示词，如果没有输入，就使用默认的问题
    std::string prompt = argc >= 4 ? argv[3] : "What is AI?";
    
    // 步数上限，含 prompt 阶段，默认是 2560
    int max_steps = argc >= 5 ? std::atoi(argv[4]) : 2560;

    model::Qwen3Model model(tokenizer_path, checkpoint_path);
    
    // 加载权重并初始化显存缓冲区
    auto init_status = model.init();
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: "
                   << init_status.get_err_code();
    }

    std::cout << "Q: " << prompt << std::endl;
    std::cout << "A: ";
    fflush(stdout);
    
    // 包装成 ChatML 对话模板
    const std::string &formatted_prompt = fill_template(prompt);
    
    // 执行推理，并记录耗时
    auto start = std::chrono::steady_clock::now();
    int steps = generate(model, formatted_prompt, max_steps, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    
    // 本次对话的统计信息
    printf("\n[steps:%d duration:%.3lfs speed:%.3lf tokens/s]\n",
        steps, duration, static_cast<double>(steps) / duration);
    fflush(stdout);
    
    return 0;
}
