# InferLite

Qwen3-0.6B 的 CUDA FP32 推理。

- 仅支持 Qwen3-0.6B。加载权重文件时校验结构参数，不匹配即报错
- 仅支持 CUDA。主机内存只用于读权重、token id 与采样回读
- 解码为 argmax；分词器直接解析 HuggingFace `tokenizer.json`
- 权重格式：本仓库自定的 FP32 二进制，由导出脚本生成

## 目录结构

```
demo/                   推理程序(infer)
InferLite/
  include/              头文件(base / model / op / tensor / tokenizer)
  src/                  实现(CUDA kernel 与算子)
models/                 运行时需要的文件(qwen0.6.bin、tokenizer.json)
tools/                  权重导出脚本
```

## 导出权重

权重文件不在仓库里，需先从 HuggingFace 下载 Qwen3-0.6B（一次性），再导出：

```bash
# 下载 HuggingFace 原始模型(一次性),放在当前目录下的 Qwen3-0.6B/
huggingface-cli download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B

python3 tools/export_qwen3.py \
  --model_dir Qwen3-0.6B \
  --output models/qwen0.6.bin
# 分词器也要放进 models/
cp Qwen3-0.6B/tokenizer.json models/
```

脚本只依赖 numpy，直接读 `model.safetensors` 与 `config.json`，不需要 torch。产物约 3GB；`--max_seq_len` 控制文件头中的序列长度上限，也就是 KV cache 容量（默认 4096）。

## 构建与运行

依赖：CMake 3.24+、C++20 编译器、CUDA Toolkit、glog、re2、nlohmann/json。

```bash
cmake -S . -B build
cmake --build build -j
./build/demo/infer models/qwen0.6.bin models/tokenizer.json \
  "What is AI?" 2560
```

后两个参数可省略（默认问题 `What is AI?`、默认步数 2560）。

若 re2 或 nlohmann/json 不在系统默认路径，configure 会报 `Could not find RE2_INCLUDE_DIR` 之类的错，用 `-D` 补路径即可：

```bash
cmake -S . -B build \
  -DRE2_INCLUDE_DIR=<re2 头文件目录> \
  -DRE2_LIBRARY=<libre2.so 路径> \
  -DNLOHMANN_JSON_INCLUDE_DIR=<json.hpp 所在目录>
```

## 权重文件格式

文件头为 8 个 int32：`dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len, intermediate_dim`（对应 `InferLite/include/model/config.h` 的 `ModelConfig`）。其后按固定顺序平铺全部 FP32 权重，顺序与 `tools/export_qwen3.py` 中 `build_weight_keys` 一致，加载顺序在 `InferLite/src/model/qwen3.cpp` 的 `create_param_layers`。
