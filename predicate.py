import mindspore as ms
import numpy as np
from mindformers import LlamaForCausalLM, LlamaConfig
from mindformers.tools.register import MindFormerConfig
from transformers import AutoTokenizer as HFTokenizer

# ================= 配置区 =================
yaml_path = "/root/output/finetune_qwen2_sparktss_lora_autogen.yaml"
ckpt_path = "/root/output/output_sparktss_lora/checkpoint/rank_0/spark_tts_rank_0-8490_1.safetensors"
TOKENIZER_DIR = "/root/LLM"
# =========================================

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# ------------------------------------------------------------------
# 第一步：加载模型
# ------------------------------------------------------------------
print("1. 正在解析配置...")
raw_config = MindFormerConfig(yaml_path)
if 'model' in raw_config and 'model_config' in raw_config['model']:
    model_config_args = raw_config['model']['model_config']
else:
    model_config_args = raw_config

# 手动修正配置
model_config_args['checkpoint_name_or_path'] = None
model_config_args['use_past'] = False     # 关闭 KV Cache 避免算子报错
model_config_args['rotary_dtype'] = "bfloat16" # 修正精度匹配

print("2. 正在初始化模型架构...")
config = LlamaConfig(**model_config_args)
model = LlamaForCausalLM(config)

print(f"3. 正在手动加载权重: {ckpt_path}")
param_dict = ms.load_checkpoint(ckpt_path, format='safetensors')

print("4. 正在注入权重...")
param_not_load, _ = ms.load_param_into_net(model, param_dict)
real_missing = [k for k in param_not_load if "cache" not in k]
if not real_missing:
    print(">>> 权重加载成功！")
else:
    print(f">>> 警告: 丢失部分权重: {real_missing[:5]}...")

# ------------------------------------------------------------------
# 第二步：加载分词器
# ------------------------------------------------------------------
print(f"5. 正在加载分词器...")
try:
    tokenizer = HFTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False, trust_remote_code=True)
    print(">>> 分词器加载成功！")
except Exception as e:
    print(f"❌ 分词器加载失败: {e}")
    raise e

# ------------------------------------------------------------------
# 第三步：推理生成
# ------------------------------------------------------------------
input_text = "你好，这是一次全量微调后的语音合成测试。"
prompt = f"<|task_tts|><|start_content|>{input_text}<|end_content|><|start_global_token|>"
print(f"\n输入 Prompt: {prompt}")

print("6. 开始生成 (use_past=False 模式)...")

# 文本转 ID
inputs = tokenizer(prompt, return_tensors="np")
input_ids_np = inputs["input_ids"]
input_ids_ms = ms.Tensor(input_ids_np, ms.int32)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# 准备 eos_token_id 列表
eos_id = tokenizer.eos_token_id
if isinstance(eos_id, int):
    eos_id = [eos_id]

# 模型生成
output_ids = model.generate(
    input_ids_ms,
    max_new_tokens=2048,
    do_sample=True,
    top_k=5,
    top_p=0.85,
    repetition_penalty=1.05,
    eos_token_id=eos_id,  # 【关键修复】必须是列表
    pad_token_id=pad_id
)

output_ids_list = output_ids[0].tolist()
output_text = tokenizer.decode(output_ids_list, skip_special_tokens=False)

print("\n" + "="*20 + " 生成结果 " + "="*20)
print(output_text)

with open("result_tokens.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
print("\n结果已保存至 result_tokens.txt")