import mindspore as ms
import os

# ================= 配置区 =================
INPUT_CKPT = "/root/LLM/mindspore_model.ckpt"
# 最终修正版 ckpt
OUTPUT_CKPT = "/root/LLM/mindspore_model_final.ckpt"


# =========================================

def patch():
    print(f">>> 正在读取: {INPUT_CKPT}")
    if not os.path.exists(INPUT_CKPT):
        print("❌ 文件不存在")
        return

    param_dict = ms.load_checkpoint(INPUT_CKPT)
    new_params = []

    # 标记是否找到了 embedding，以便复制给 lm_head
    embed_tensor = None
    has_lm_head = False
    has_norm_out = False

    print(">>> 正在应用补丁...")
    for name, tensor in param_dict.items():
        # 1. 修复 Norm 层名字
        if name == "model.norm.weight":
            print(f"   🛠️ 修复: {name} -> model.norm_out.weight")
            name = "model.norm_out.weight"
            has_norm_out = True
        elif name == "model.norm_out.weight":
            has_norm_out = True

        # 记录 Embedding 用于克隆
        if name == "model.tok_embeddings.embedding_weight":
            embed_tensor = tensor

        # 检查是否已有 lm_head
        if name == "lm_head.weight":
            has_lm_head = True

        new_params.append({"name": name, "data": tensor})

    # 2. 修复 LM Head (如果缺失，从 Embedding 克隆)
    if not has_lm_head:
        if embed_tensor is not None:
            print("   🛠️ 修复: 缺失 lm_head.weight，正在从 Embedding 克隆...")
            new_params.append({"name": "lm_head.weight", "data": embed_tensor})
        else:
            print("❌ 严重错误: 没找到 Embedding 层，无法克隆 lm_head！")

    if not has_norm_out:
        print("⚠️ 警告: 没找到 model.norm.weight，可能名字不对？")

    print(f">>> 正在保存最终版: {OUTPUT_CKPT}")
    ms.save_checkpoint(new_params, OUTPUT_CKPT)
    print("✅ 补丁完成！")


if __name__ == "__main__":
    patch()