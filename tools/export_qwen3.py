#!/usr/bin/env python3
"""把 HuggingFace 上的 Qwen3-0.6B 权重导出为 InferLite 的 FP32 二进制格式。

直接从 model.safetensors + config.json 读取,不需要 torch / transformers。
输出文件由一个 8×int32 的文件头(ModelConfig,与 InferLite/include/model/config.h
对应)加一串 float32 权重组成,权重顺序与 C++ 侧加载顺序
(InferLite/src/model/qwen3.cpp 的 create_param_layers)严格一致。

用法:
    python3 tools/export_qwen3.py \
        --model_dir <HuggingFace 模型目录> --output models/qwen0.6.bin

依赖: numpy(pip install numpy 或 uv run --with numpy 运行本脚本)。
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# Qwen3-0.6B 架构常量,与 C++ 侧 generate_model_infos 的校验一致。
EXPECTED_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "intermediate_size": 3072,
}


def load_safetensors_header(path):
    """读取 safetensors 文件头,返回 (header, data_start)。

    data_start 是张量数据区在文件中的起始偏移;每个张量的
    data_offsets 是相对 data_start 的偏移。
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def read_tensor_fp32(path, info, data_start):
    """按 data_offsets 读取单个张量并转为 float32 一维数组。"""
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    dtype = info["dtype"]
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if dtype == "BF16":
        # bf16 -> fp32 为精确移位,直接搬位即可。
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    raise ValueError(f"不支持的 safetensors 数据类型: {dtype}")


def build_weight_keys(layer_num):
    """按 C++ 消费顺序生成权重键列表(每组张量按层号 0..N 排列)。"""
    keys = []
    for i in range(layer_num):
        keys.append(f"model.layers.{i}.input_layernorm.weight")
    for i in range(layer_num):
        keys.append(f"model.layers.{i}.post_attention_layernorm.weight")
    keys.append("model.norm.weight")
    keys.append("model.embed_tokens.weight")
    # 注意力投影:q/k/v 与各自的 q_norm/k_norm,随后 o
    for proj in ["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"]:
        for i in range(layer_num):
            keys.append(f"model.layers.{i}.self_attn.{proj}.weight")
    # FFN:gate/down/up 与 PyTorch 中的 w1/w2/w3 一一对应
    for proj in ["gate_proj", "down_proj", "up_proj"]:
        for i in range(layer_num):
            keys.append(f"model.layers.{i}.mlp.{proj}.weight")
    keys.append("lm_head.weight")
    return keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-0.6B HuggingFace weights to the InferLite FP32 binary format.")
    parser.add_argument("--model_dir", type=Path, default=Path("Qwen3-0.6B"),
                        help="HuggingFace 模型目录(含 model.safetensors 与 config.json)")
    parser.add_argument("--output", type=Path, default=Path("models/qwen0.6.bin"),
                        help="输出文件路径")
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="写入文件头的最大序列长度(限制 KV cache 显存占用),"
                             "默认取 4096;不会超过模型的 max_position_embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    st_path = args.model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {st_path}")
    config = json.loads((args.model_dir / "config.json").read_text())
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"模型结构不是 Qwen3-0.6B(config.{key}={config.get(key)}, 期望 {expected})。"
                f"本项目只支持 Qwen3-0.6B。")

    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    seq_len = min(config["max_position_embeddings"], args.max_seq_len)
    header = struct.pack("iiiiiiii",
                         config["num_attention_heads"] * head_dim,  # dim
                         config["hidden_size"],
                         config["num_hidden_layers"],
                         config["num_attention_heads"],
                         config["num_key_value_heads"],
                         config["vocab_size"],
                         seq_len,
                         config["intermediate_size"])

    keys = build_weight_keys(config["num_hidden_layers"])
    st, data_start = load_safetensors_header(st_path)
    missing = [k for k in keys if k not in st]
    if missing:
        raise KeyError(f"权重文件中缺少张量: {missing[:5]} ...")

    with open(args.output, "wb") as out:
        out.write(header)
        for key in keys:
            weight = read_tensor_fp32(st_path, st[key], data_start)
            if weight.size == 0:
                raise ValueError(f"张量为空: {key}")
            out.write(weight.astype(np.float32, copy=False).tobytes())
    size_gb = args.output.stat().st_size / 1e9
    print(f"已导出 {len(keys)} 个权重张量 -> {args.output} ({size_gb:.2f} GB, fp32)")


if __name__ == "__main__":
    main()
