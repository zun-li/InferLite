# InferLite

Qwen3-0.6B 的 CUDA FP32 推理。

## 特性

- 完整 Qwen3 结构：RMSNorm（含 Q/K-Norm）、RoPE、GQA（16 个 Q head / 8 个 KV head）、SwiGLU FFN、残差连接
- 所有算子手写 CUDA kernel：matmul / rmsnorm / rope / swiglu / mha / embedding / add / argmax
- FP32 KV cache，按层预分配，容量等于导出时的 `seq_len`
- 权重文件 mmap 到主机内存，各算子在初始化时 `cudaMemcpy` 进显存，全流程无需 torch / transformers

## 限制

- 仅支持 Qwen3-0.6B。加载权重文件时校验结构参数，不匹配即报错
- 仅支持 CUDA。主机内存只用于读权重、token id 与采样回读
- 解码为 argmax（贪心）；分词器直接解析 HuggingFace `tokenizer.json`
- 无批处理、无量化、无 CPU 回退
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

`models/` 被 `.gitignore` 忽略，克隆后必须先按下文导出权重，不能直接运行。

## 依赖

| 依赖 | 说明 |
| --- | --- |
| CMake | 3.24+ |
| C++ 编译器 | 需支持 C++20 |
| CUDA Toolkit | 提供 `nvcc` 与 `cudart`，C++ 与 CUDA 均按 C++20 编译 |
| NVIDIA GPU + 驱动 | 显存 ≥ 4GB。权重约 2.8GiB，KV cache 约 0.9GiB（`seq_len=4096` 时），另需 CUDA 上下文开销 |
| glog | 系统前缀，`find_package(glog REQUIRED)` |
| re2、nlohmann/json | 以 `find_package(... CONFIG)` 定位 |
| Python 3 + numpy | 仅导出权重时需要 |

只有 CUDA 后端，没有 CPU 回退。CUDA 目标架构默认取本机 GPU（`CMAKE_CUDA_ARCHITECTURES=native`）；要在别的机器上运行，configure 时需显式指定，例如 `-DCMAKE_CUDA_ARCHITECTURES=89`。

## 导出权重

权重文件不在仓库里，需先从 HuggingFace 下载 Qwen3-0.6B（一次性），再导出：

```bash
# 1. 安装导出脚本的依赖(仅 numpy)
pip install numpy

# 2. 下载 HuggingFace 原始模型(一次性),放在当前目录下的 Qwen3-0.6B/
hf download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
# 旧版 huggingface_hub 可改用:
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B

# 3. 导出推理权重,并把分词器也放进 models/
python3 tools/export_qwen3.py \
  --model_dir Qwen3-0.6B \
  --output models/qwen0.6.bin
cp Qwen3-0.6B/tokenizer.json models/
```

脚本只依赖 numpy，直接读 `model.safetensors` 与 `config.json`，不需要 torch。产物约 3GB；`--max_seq_len` 控制文件头中的序列长度上限，也就是 KV cache 容量（默认 4096）。

## 构建

glog 走系统前缀，re2 与 nlohmann/json 通过 `find_package(... CONFIG)` 定位。若这两个包装在非默认前缀（常见于 conda），**configure 必须用 `CMAKE_PREFIX_PATH` 指定**，否则会以
`Could not find a package configuration file provided by "re2"` 直接失败：

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/.local/miniconda3
cmake --build build -j
```

两者都在系统默认前缀时，可以省略该参数：

```bash
cmake -S . -B build
cmake --build build -j
```

## 运行

```bash
./build/demo/infer models/qwen0.6.bin models/tokenizer.json \
  "What is AI?" 2560
```

参数全部按位置传入：

| 位置 | 含义 | 默认值 |
| --- | --- | --- |
| `argv[1]` | 权重文件 | 必填 |
| `argv[2]` | `tokenizer.json` | 必填 |
| `argv[3]` | prompt | `What is AI?` |
| `argv[4]` | 步数上限 | 2560 |

后两个参数可省略。步数上限含 prompt 阶段：prompt 先逐 token 建立 KV cache，之后每步解码一个 token，命中 `<|im_end|>` 或 `<|endoftext|>` 提前结束。

## 示例输出

用默认 prompt 在 RTX 4060 上运行：

```bash
./build/demo/infer models/qwen0.6.bin models/tokenizer.json
```

```text
Q: What is AI?
A: <think>
Okay, the user is asking what AI is. I need to explain it clearly. Let me start by defining AI broadly. It's artificial intelligence, which is a subset of computer science. I should mention that it's about machines learning from data.

Wait, the user might not know the basics. Maybe I should break it down into parts. Like, what AI does, how it works, and examples. Also, note that AI is a field of study, not just a technology. Make sure to keep it simple and avoid jargon.

I should check if there's any recent developments or applications that the user might be interested in. But since the question is general, maybe stick to the basics. Also, mention that AI can be used in various areas like healthcare, finance, etc. That should cover it.
</think>

Artificial Intelligence (AI) refers to the simulation of human intelligence tasks performed by machines. It involves the development of algorithms and models that can learn from data, make decisions, and solve problems. AI is a field of computer science that focuses on creating systems capable of performing tasks that would typically require human intelligence, such as understanding language, recognizing patterns, and making predictions.

AI works by analyzing data, identifying patterns, and learning from past experiences to improve performance over time. This learning process is achieved through machine learning, where models are trained on large datasets to make accurate predictions or decisions. AI has a wide range of applications, including healthcare (diagnosis, personalized treatment), finance (risk assessment, fraud detection), and autonomous systems (self-driving cars, recommendation engines).

AI is not limited to just technology; it is a broad field that spans various domains and continues to evolve rapidly.
[steps:355 duration:4.159s speed:85.366 tokens/s]
```

`steps` 是实际执行的步数（含 prompt 阶段），最后一行同时给出耗时与吞吐。解码是 argmax，所以 `steps` 与正文可复现，`duration` / `speed` 随机器负载浮动。

`<think>` 块是模型自己生成的正文：`demo/main.cpp` 的对话模板没有关闭 thinking，程序也只过滤 `<|im_start|>` / `<|im_end|>` 两个模板符，因此思考内容会原样打印。

## 权重文件格式

整个文件 = 32 字节文件头 + 连续的 FP32 权重数据，没有对齐填充和额外元数据：

```text
offset 0                     offset 32                                file end
+----------------------------+----------------------------------------------+
|  header: 8 x int32 (32 B)  |  weights: float32, contiguous                |
|                            |  tensor #0, #1, ... , #(n-1)                 |
+----------------------------+----------------------------------------------+
```

### 文件头

按偏移顺序排列的 8 个 int32，字段与 `ModelConfig`（`InferLite/include/model/config.h`）一一对应：

| 偏移 | 字段 | 含义 |
| ---: | --- | --- |
| 0 | `dim` | `head_num × head_size` |
| 4 | `hidden_dim` | 隐藏层维度 |
| 8 | `layer_num` | Transformer 层数 |
| 12 | `head_num` | Q head 数 |
| 16 | `kv_head_num` | KV head 数 |
| 20 | `vocab_size` | 词表大小 |
| 24 | `seq_len` | 序列长度上限，即 KV cache 容量 |
| 28 | `intermediate_dim` | FFN 中间维度 |

不做字节序转换，按写入机器的本机字节序（x86_64 上为小端）。

### 权重数据

从偏移 32 开始是全部权重的 FP32 原始数据，没有名字、没有长度前缀、没有对齐。张量**按种类分组**，同一组内部按层号 `i = 0 .. N-1` 排列：

| 组 | 内容 | 张量数 |
| ---: | --- | ---: |
| 1 | `model.layers.{i}.input_layernorm.weight` | N |
| 2 | `model.layers.{i}.post_attention_layernorm.weight` | N |
| 3 | `model.norm.weight` | 1 |
| 4 | `model.embed_tokens.weight` | 1 |
| 5 | `self_attn` 的 `q_proj → q_norm → k_proj → k_norm → v_proj → o_proj`，每个投影各遍历一遍所有层 | 6N |
| 6 | `mlp` 的 `gate_proj → down_proj → up_proj`，同上 | 3N |
| 7 | `lm_head.weight` | 1 |

合计 `11N + 3` 个张量（Qwen3-0.6B 的 `N = 28`，即 311 个）。生成顺序见 `tools/export_qwen3.py` 的 `build_weight_keys()`，消费顺序见 `InferLite/src/model/qwen3.cpp` 的 `create_param_layers()`，两者必须严格一致。

以 Qwen3-0.6B、`--max_seq_len 4096` 为例，文件头这 8 个 int32 是：

```text
2048, 1024, 28, 16, 8, 151936, 4096, 3072
```

其中 `seq_len` 取 `min(config.max_position_embeddings, --max_seq_len)`，其余字段直接来自 HuggingFace 的 `config.json`。
