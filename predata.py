import os

# 控制线程数，避免直接爆炸了
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import csv
import json
from tqdm import tqdm
import argparse
import sys
import torch

sys.path.append("/root/SparkTTSmain")

from SparkTTSmain.sparktts.models.audio_tokenizer import BiCodecTokenizer


class AudioPromptDataset:
    def __init__(self, model_name_or_path, device):
        self.audio_tokenizer = BiCodecTokenizer(model_name_or_path, device=device)

    def tokenize(self, data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # 关键修复1：标准化 data_dir 路径（处理结尾是否带 /，统一分隔符）
        data_dir = os.path.normpath(data_dir)
        output_file = os.path.join(output_dir, os.path.basename(data_dir) + ".jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            metadata_path = os.path.join(data_dir, "metadata.csv")
            # 关键修复2：检查 metadata.csv 是否存在
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"找不到 metadata.csv：{metadata_path}")

            with open(metadata_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter=",")
                next(reader)  # Skip header

                for row in tqdm(reader):
                    if len(row) >= 2:
                        audio_path, text = row[:2]
                    else:
                        print(f"[WARNING] Skipping invalid row: {row}")
                        continue

                    # -------------------------- 核心修复：路径标准化 --------------------------
                    # 1. 替换反斜杠为正斜杠（处理 Windows 风格路径）
                    audio_path = audio_path.replace("\\", "/")
                    # 2. 清理重复斜杠（比如 "wav//62.wav" → "wav/62.wav"）
                    audio_path = "/".join([part for part in audio_path.split("/") if part])
                    # 3. 拼接路径并标准化（自动适配系统分隔符）
                    audio_path = os.path.normpath(os.path.join(data_dir, audio_path))
                    # ----------------------------------------------------------------------

                    # 关键：打印路径，确认是否正确（调试用，可后续删除）
                    if not os.path.exists(audio_path):
                        print(f"[WARNING] 音频文件不存在：{audio_path} → 跳过该样本")
                        continue

                    try:
                        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(audio_path)
                    except Exception as e:
                        print(f"[ERROR] 处理音频 {audio_path} 失败：{e} → 跳过该样本")
                        continue

                    global_tokens = "".join(
                        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
                    )
                    semantic_tokens = "".join(
                        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
                    )

                    inputs = [
                        "<|task_tts|>",
                        "<|start_content|>", text, "<|end_content|>",
                        "<|start_global_token|>", global_tokens, "<|end_global_token|>",
                        "<|start_semantic_token|>", semantic_tokens, "<|end_semantic_token|>",
                        "<|im_end|>"
                    ]

                    prompt = {"text": "".join(inputs)}
                    f.write(json.dumps(prompt, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio dataset for TTS.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="output_prompt", help="Path to save the tokenized output.")

    args = parser.parse_args()

    processor = AudioPromptDataset(
        model_name_or_path="SparkTTSmain",
        device="cpu"
    )

    processor.tokenize(args.data_dir, args.output_dir)