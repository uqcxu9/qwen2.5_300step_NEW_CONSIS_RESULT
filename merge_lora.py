#!/usr/bin/env python3
"""
Merge LoRA into base model and save a standalone model.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 到 base model")
    parser.add_argument('--base_model', type=str, required=True, help='Base model 路径')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA 权重路径 (adapter 目录)')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    if not os.path.exists(args.base_model):
        print(f"❌ ERROR: Base model 不存在: {args.base_model}")
        sys.exit(1)
    if not os.path.exists(args.lora_path):
        print(f"❌ ERROR: LoRA 路径不存在: {args.lora_path}")
        sys.exit(1)

    adapter_cfg = os.path.join(args.lora_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        print(f"❌ ERROR: {args.lora_path} 里没有 adapter_config.json（这不是 LoRA adapter 目录）")
        sys.exit(1)

    torch.manual_seed(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Output path: {args.output_path}")

    # ✅ 强制在 CPU 加载/合并，避免 OOM
    print("\n加载 base model (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )

    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("合并 LoRA 到 base model...")
    model.eval()
    with torch.no_grad():
        model = model.merge_and_unload()

    print(f"保存到 {args.output_path} ...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    print("保存 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print("\n✅ 合并完成！建议你检查输出目录是否包含 config.json / *.safetensors / tokenizer 文件。")
    print("\n运行 simulation 示例：")
    print("  cd /workspace/QWEN2.5_42_GRPO_700step-/QWEN2.5_42_GRPO_1")
    print(f"  python simulate.py --policy_model qwen --model_type qwen --num_agents 100 --episode_length 240 --seed {args.seed}")

if __name__ == "__main__":
    main()
