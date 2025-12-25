import torch
import soundfile as sf
import json
import os
import sys
import re
from safetensors.torch import load_file  # 需要 pip install safetensors

# ================= 配置区 =================
# 1. Spark-TTS 源码路径 (必须设置，用于导入 BiCodec 类)
SPARK_TTS_CODE_DIR = "/root/Spark-TTS-main"

# 2. 模型文件夹路径 (该目录下必须有 model.safetensors 和 config.json)
BICODEC_DIR = "./Bicodec"
MODEL_FILENAME = "model.safetensors"  # 你的文件名

# 3. Token 文件
TOKEN_FILE = "generated_tokens.txt"
# =========================================

# 导入 Spark-TTS 源码
sys.path.append(SPARK_TTS_CODE_DIR)
try:
    from spark_tts.model.audio_codec import BiCodec

    print(">>> 成功导入 BiCodec 类")
except ImportError:
    print(f"❌ 无法导入 Spark-TTS 代码，请检查 SPARK_TTS_CODE_DIR: {SPARK_TTS_CODE_DIR}")
    exit()


def extract_ids(text):
    g_ids = [int(x) for x in re.findall(r'\|bicodec_global_(\d+)\|', text)]
    s_ids = [int(x) for x in re.findall(r'\|bicodec_semantic_(\d+)\|', text)]
    return g_ids, s_ids


def main():
    device = "cpu"
    config_path = os.path.join(BICODEC_DIR, "config.json")
    ckpt_path = os.path.join(BICODEC_DIR, MODEL_FILENAME)

    # 1. 检查文件
    if not os.path.exists(config_path):
        print(f"❌ 错误：找不到配置文件 {config_path}")
        print("Safetensors 格式只存了参数，由于不知道模型结构，必须要有 config.json 才能初始化模型！")
        return

    # 2. 初始化模型结构
    print(">>> 正在初始化 BiCodec 模型结构...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 实例化模型 (根据 config)
    model = BiCodec(**config)

    # 3. 加载 Safetensors 权重
    print(f">>> 正在加载权重: {ckpt_path}")
    try:
        state_dict = load_file(ckpt_path)  # 使用 safetensors 库加载
        model.load_state_dict(state_dict)  # 注入权重
        model.to(device)
        model.eval()
        print(">>> ✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    # 4. 读取 Token 并解码
    if not os.path.exists(TOKEN_FILE):
        print(f"❌ 找不到 {TOKEN_FILE}")
        return

    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    g_ids, s_ids = extract_ids(text)
    print(f"提取到: Global={len(g_ids)}, Semantic={len(s_ids)}")

    if not s_ids:
        print("❌ 未提取到 Semantic Tokens，无法生成。")
        return

    print(">>> 正在解码音频...")
    with torch.no_grad():
        # 构造输入张量
        semantic_tensor = torch.tensor([s_ids], dtype=torch.long, device=device)
        global_tensor = torch.tensor([g_ids], dtype=torch.long, device=device)

        # 解码
        wav = model.decode(semantic_tensor, global_tensor)

    save_path = "final_output.wav"
    sf.write(save_path, wav.squeeze().numpy(), 24000)
    print(f"\n🎉 恭喜！音频已生成: {save_path}")
    print("快下载下来听听吧！")


if __name__ == "__main__":
    main()