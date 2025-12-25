import os, json, subprocess
import numpy as np

# =========================
# 你需要改的参数
# =========================
MODEL_DIR = "./LLM"
TRAIN_JSONL = "./data/data.jsonl"
WORK_DIR = "./output"
REGISTER_PATH = "research/qwen2_5"

# [关键修改 1] 必须加大序列长度，Spark-TTS 音频很长
SEQ_LEN = 2048
# 如果显存不够(OOM)，可尝试降为 2048，但绝对不能是 256

BATCH_SIZE = 1
GRAD_ACC = 8  # 累积步数根据 SEQ_LEN 增大适当调整
EPOCHS = 10  # 建议多跑几轮
LR = 1e-4  # [建议] LoRA 学习率调大一点
IGNORE_ID = -100

SPLIT_TOKEN = "<|start_global_token|>"

MR_PATH = os.path.join(WORK_DIR, "sparktss_train.mindrecord")
YAML_PATH = os.path.join(WORK_DIR, "finetune_qwen2_sparktss_lora_autogen.yaml")
# 注意：YAML 中的 output_dir 最好是绝对路径，避免找不到
OUT_DIR = os.path.join(WORK_DIR, "output_sparktss_lora")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) 生成 MindRecord
# =========================
if not os.path.exists(MR_PATH):
    print(f"[1/3] build mindrecord (Seq Len: {SEQ_LEN}) ...")
    from transformers import AutoTokenizer
    from mindspore.mindrecord import FileWriter

    # 确保加载时不要限制 vocab size
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)

    # 强制修正 pad_id，Qwen 有时默认无 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pad_id = tokenizer.pad_token_id
    split_id = tokenizer.convert_tokens_to_ids(SPLIT_TOKEN)

    writer = FileWriter(MR_PATH, shard_num=1)
    # [关键修改 2] 补全 Schema，增加 attention_mask
    writer.add_schema({
        "input_ids": {"type": "int32", "shape": [SEQ_LEN]},
        "labels": {"type": "int32", "shape": [SEQ_LEN]},
        "attention_mask": {"type": "int32", "shape": [SEQ_LEN]},
    }, "sparktss_lm")

    rows = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            full = obj["text"] if "text" in obj else obj.get("input", "") + obj.get("output", "")

            # 编码
            ids = tokenizer(full, add_special_tokens=False).input_ids

            # 找分割点
            try:
                start_pos = ids.index(split_id)
            except ValueError:
                # 没找到分割点（只有文本？），全部忽略 loss
                start_pos = len(ids)

            # 构建 Labels：文本部分设为 -100，音频部分保留 ID
            labels = [IGNORE_ID] * start_pos + ids[start_pos:]

            # 构建 Mask：有效部分为 1
            attn = [1] * len(ids)

            # 截断 (Truncation)
            ids = ids[:SEQ_LEN]
            labels = labels[:SEQ_LEN]
            attn = attn[:SEQ_LEN]

            # 填充 (Padding)
            if len(ids) < SEQ_LEN:
                pad_len = SEQ_LEN - len(ids)
                ids += [pad_id] * pad_len
                labels += [IGNORE_ID] * pad_len  # Pad 部分也不算 Loss
                attn += [0] * pad_len  # Pad 部分 Mask 为 0

            # [关键修改 3] 写入所有字段
            rows.append({
                "input_ids": np.asarray(ids, np.int32),
                "labels": np.asarray(labels, np.int32),
                "attention_mask": np.asarray(attn, np.int32),
            })

    writer.write_raw_data(rows)
    writer.commit()
    print("mindrecord saved:", MR_PATH)
else:
    print("[1/3] mindrecord exists, skip build. (Delete it if you changed SEQ_LEN)")

# =========================
# 2) 自动生成 LoRA finetune YAML
# =========================
print("[2/3] write yaml ...")

# 格式化字符串中的 { } 需要转义为 {{ }}
yaml_text = f"""
seed: 0
output_dir: "{OUT_DIR}"
run_mode: "finetune"

# 加载权重
load_checkpoint: "{MODEL_DIR}/model.safetensors"
load_ckpt_format: "safetensors"
auto_trans_ckpt: False
use_parallel: False

trainer:
  type: CausalLanguageModelingTrainer
  model_name: "qwen2_05b"

runner_config:
  epochs: {EPOCHS}
  batch_size: {BATCH_SIZE}
  sink_mode: True
  sink_size: 1
  gradient_accumulation_steps: {GRAD_ACC}

optimizer:
  type: AdamW
  learning_rate: {LR}
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01

lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: {LR}
  lr_end: 1.e-6
  warmup_ratio: 0.03
  total_steps: -1

train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{MR_PATH}"
    shuffle: True
  # [关键修改 4] 列名必须与 MindRecord Schema 严格一致
  input_columns: ["input_ids", "labels", "attention_mask"]
  drop_remainder: True
  batch_size: {BATCH_SIZE}
  repeat: 1

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset

context:
  mode: 0
  device_target: "Ascend"
  device_id: 0
  max_device_memory: "25GB"

callbacks:
  - type: MFLossMonitor
  - type: CheckpointMonitor
    prefix: "spark_tts"
    save_checkpoint_steps: 500
    keep_checkpoint_max: 2
    integrated_save: False
    async_save: False
    checkpoint_format: "safetensors"

model:
  model_config:
    type: LlamaConfig
    seq_length: {SEQ_LEN}

    # [关键修改 5] 必须显式指定扩充后的词表大小
    vocab_size: 166000

    hidden_size: 896
    num_layers: 24
    num_heads: 14
    n_kv_heads: 2
    intermediate_size: 4864
    qkv_has_bias: True
    rms_norm_eps: 1.0e-6
    rope_theta: 1000000.0

    use_past: False
    use_flash_attention: True
    compute_dtype: "bfloat16"
    layernorm_compute_type: "float32"
    softmax_compute_type: "float32"
    rotary_dtype: "float32"

    pet_config:
      pet_type: "lora"
      lora_rank: 32
      lora_alpha: 64
      lora_dropout: 0.05
      # [建议] 扩充 LoRA 范围
      target_modules: ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj|.*wq|.*wk|.*wv|.*wo"

  arch:
    type: LlamaForCausalLM
  processor:
    return_tensors: ms
    tokenizer:
      vocab_file: "{MODEL_DIR}/vocab.json"
      merges_file: "{MODEL_DIR}/merges.txt"
"""

with open(YAML_PATH, "w", encoding="utf-8") as f:
    f.write(yaml_text.strip() + "\n")

print("yaml saved:", YAML_PATH)