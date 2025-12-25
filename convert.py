import torch
import mindspore as ms
from safetensors.torch import load_file
import os

# ================= 配置区 =================
# 输入: 原始 HF 权重 (safetensors)
INPUT_PATH = "/root/LLM/model.safetensors"
# 输出: 转换后的 MS 权重 (ckpt)
OUTPUT_PATH = "/root/LLM/mindspore_model.ckpt"


# =========================================

def map_name(name):
    """
    将 HF Qwen2 参数名映射到 MindFormers Llama 参数名
    """
    # 1. 全局 Embedding 和 Output
    if name == "model.embed_tokens.weight":
        return "model.tok_embeddings.embedding_weight"
    if name == "lm_head.weight":
        return "lm_head.weight"
    if name == "model.norm.weight":
        return "model.norm.weight"

    # 2. Attention 层
    # Qwen: model.layers.0.self_attn.q_proj.weight/bias
    # MF  : model.layers.0.attention.wq.weight/bias
    if "self_attn.q_proj" in name:
        return name.replace("self_attn.q_proj", "attention.wq")
    if "self_attn.k_proj" in name:
        return name.replace("self_attn.k_proj", "attention.wk")
    if "self_attn.v_proj" in name:
        return name.replace("self_attn.v_proj", "attention.wv")
    if "self_attn.o_proj" in name:
        return name.replace("self_attn.o_proj", "attention.wo")

    # 3. MLP 层 (Feed Forward)
    # Qwen: gate_proj -> w1, down_proj -> w2, up_proj -> w3
    # 注意：MindFormers Llama 通常定义 w1=gate, w2=down, w3=up
    if "mlp.gate_proj" in name:
        return name.replace("mlp.gate_proj", "feed_forward.w1")
    if "mlp.down_proj" in name:
        return name.replace("mlp.down_proj", "feed_forward.w2")
    if "mlp.up_proj" in name:
        return name.replace("mlp.up_proj", "feed_forward.w3")

    # 4. LayerNorm 层
    # Qwen: input_layernorm -> attention_norm
    # Qwen: post_attention_layernorm -> ffn_norm
    if "input_layernorm" in name:
        return name.replace("input_layernorm", "attention_norm")
    if "post_attention_layernorm" in name:
        return name.replace("post_attention_layernorm", "ffn_norm")

    # 如果没匹配上，打印出来
    return None


def main():
    print(f">>> 正在加载 HuggingFace 权重: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ 文件不存在")
        return

    try:
        hf_state = load_file(INPUT_PATH)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    ms_params = []
    print(">>> 开始转换参数名...")

    count = 0
    for name, tensor in hf_state.items():
        ms_name = map_name(name)

        if ms_name is None:
            print(f"⚠️ 警告: 无法映射参数 {name}，将被丢弃")
            continue

        # 打印前几个转换示例
        if count < 5:
            print(f"   [映射] {name} -> {ms_name}")
            count += 1

        # 转换数据类型: bfloat16 -> float32
        # (MindSpore ckpt 建议保存为 float32，加载时框架会自动转精度)
        data = tensor.float().numpy()

        # 构造 Parameter
        ms_params.append({'name': ms_name, 'data': ms.Tensor(data, ms.float32)})

    print(f"\n>>> 正在保存 MindSpore Checkpoint: {OUTPUT_PATH}")
    print("    这可能需要几分钟，请耐心等待...")
    ms.save_checkpoint(ms_params, OUTPUT_PATH)
    print("✅ 转换完成！")
    print("-" * 40)
    print(f"新权重路径: {OUTPUT_PATH}")
    print("-" * 40)


if __name__ == "__main__":
    main()
