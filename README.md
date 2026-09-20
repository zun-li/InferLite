# InferLite

> Qwen3-0.6B 的 CUDA FP32 推理，全流程无需 torch / transformers。

![](./demo.gif)

## 特性

- 实现基础模块：Tensor、Buffer、Layer、Model等。
- 实现 Qwen3 结构：RMSNorm、RoPE、GQA、SwiGLU FFN、残差连接等。
- 实现 CUDA kernel：matmul / rmsnorm / rope / swiglu / mha / embedding / add / argmax等。
- FP32 KV cache，按层预分配，容量等于导出时的 `seq_len`。
- 权重文件 mmap 到主机内存，各算子在初始化时搬进显存。

## 限制

- 仅支持 Qwen3-0.6B。
- 仅支持 CUDA。
- 解码为 argmax 贪心算法。
- 暂未实现 PagedAttention、Continous Batching。

## 外部依赖

1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
5. Cuda Toolkit

## 导出权重

权重文件不在仓库里，需先从 HuggingFace 下载 Qwen3-0.6B（一次性），再导出：

```bash
# 1. 安装导出脚本的依赖
pip install numpy

# 2. 下载 HuggingFace 原始模型，放在当前目录下的 Qwen3-0.6B/
hf download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B

# 3. 导出推理权重到 models/
python3 tools/export_qwen3.py \
  --model_dir Qwen3-0.6B \
  --output models/qwen0.6.bin

# 4. 把分词器也放进 models/
cp Qwen3-0.6B/tokenizer.json models/
```

## 构建

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
./build/demo/infer models/qwen0.6.bin models/tokenizer.json "What is AI?" 2560
```

参数全部按位置传入：

| 位置 | 含义 | 默认值 |
| --- | --- | --- |
| `argv[1]` | qwen0.6.bin | **必填** |
| `argv[2]` | tokenizer.json | **必填** |
| `argv[3]` | prompt | What is AI? |
| `argv[4]` | 步数上限 | 2560 |

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

## 权重文件格式

整个文件 = 32 字节文件头 + 连续的 FP32 权重数据，没有对齐填充和额外元数据：

```text
offset 0                     offset 32                                file end
+----------------------------+----------------------------------------------+
|  header: 8 x int32 (32 B)  |  weights: float32, contiguous                |
|                            |  tensor #0, #1, ... , #(n-1)                 |
+----------------------------+----------------------------------------------+
```

### header

按偏移顺序排列的 8 个 int32，字段见 `ModelConfig`（`InferLite/include/model/config.h`）：

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

### weights

从偏移 32 开始是所有权重的 FP32 原始数据，没有名字、没有长度前缀、没有对齐。

张量**按种类分组**，同一组内部按层号 `i = 0 .. N-1` 排列：

| 组 | 内容 | 张量数 |
| ---: | --- | ---: |
| 1 | `model.layers.{i}.input_layernorm.weight` | N |
| 2 | `model.layers.{i}.post_attention_layernorm.weight` | N |
| 3 | `model.norm.weight` | 1 |
| 4 | `model.embed_tokens.weight` | 1 |
| 5 | `self_attn` 的 `q_proj → q_norm → k_proj → k_norm → v_proj → o_proj` | 6N |
| 6 | `mlp` 的 `gate_proj → down_proj → up_proj`，同上 | 3N |
| 7 | `lm_head.weight` | 1 |

合计 `11N + 3` 个张量（Qwen3-0.6B 的 `N = 28`，即 311 个）。

生成顺序见 `tools/export_qwen3.py` 的 `build_weight_keys()`。

消费顺序见 `InferLite/src/model/qwen3.cpp` 的 `create_param_layers()`。

生成顺序和消费顺序必须严格一致。

### example

以 Qwen3-0.6B、`--max_seq_len 4096` 为例，文件头这 8 个 int32 是：

```text
2048, 1024, 28, 16, 8, 151936, 4096, 3072
```

其中 `seq_len` 取 `min(config.max_position_embeddings, --max_seq_len)`。

其余字段直接来自 HuggingFace 的 `config.json`。
