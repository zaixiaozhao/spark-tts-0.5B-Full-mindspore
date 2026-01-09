# Intelligent Triage System (MindSpore)

基于 Spark-TTS-0.5B 模型与 MindSpore 框架实现的语音合成模型。

**中文简体**

---

本项目是一个基于 **Spark-TTS-0.5B** 大语言模型微调的语音合成模型。
利用 **MindSpore** 框架和 **MindNLP** 库，在 **Ascend NPU** 硬件上实现了高效的训练和推理。

## Features

1. **国产算力适配:** 完全基于华为 Ascend NPU (910B) 和 MindSpore 框架开发。
2. **全量微调:** 使用全量微调参数以适配 Spark-TTS-0.5B 的额外词表。
3. **双阶段生成架构:** 模型首先精准预测离散音频 Token ，然后利用 Bicodec 神经编解码器，将 Token 高保真还原为波形。
4. **全流程覆盖:** 提供数据转换、模型训练、自动推理的全套解决方案。

## Installation & Environments

本项目需要配置**两个独立的虚拟环境**以避免依赖冲突：

#### Install Dependencies

1. **MindSpore 训练环境:** 用于模型训练与推理（Step 2, 3, 4, 5）。
   ```bash
   pip install -r requirements2.txt
   ```
2. **BiCodec 编解码环境:** 用于数据预处理与音频解码（对应 Step 1, 6）
   ```bash
   pip install -r requirements1.txt
   ```

## Workflow

请严格按照以下顺序执行，并注意切换对应的虚拟环境。

#### Step 1: 数据转换 (Data Conversion)

从 HuggingFace 下载 Spark-TTS 模型，并保存在 SparkTTSmain 目录下
使用 predata.py 将原始数据转换为 Token 形式

```bash
python predata.py
```

#### Step 2: 生成 MindRecord (Data Processing)

使用 datatomind.py 将第一步生成的 data.jsonl 转换为 MindRecord 格式

关键配置：

* **1:** 生成的 YAML 文件默认是 LoRA 配置。

* **2:** 必须注释掉 LoRA 配置，改为全量微调。

* **原因:** 本项目使用的 Spark-TTS 模型词表在原 Qwen 词表基础上有添加，LoRA 会冻结权重导致新增词表无法更新，从而导致 Loss
  不下降。

```bash
python datatomind.py
```

#### Step 3: 模型转换 (Model Conversion)

使用 convert.py 将 HuggingFace 的 Spark-TTS 模型 (.safetensors) 转换为 MindSpore 的 Checkpoint (.ckpt)。

* **配置同步:** 请确保 Step 2 生成的 YAML 文件中，模型加载路径与本步骤生成的 mindspore_model_final.ckpt 路径一致。

```bash
python convert.py
```

#### Step 4: 训练环境搭建 (Training Setup)

拉取并配置 MindFormers 库

* **建议:** 在项目根目录下执行

```bash
git clone [https://gitee.com/mindspore/mindformers.git](https://gitee.com/mindspore/mindformers.git)
cd mindformers
# 关键: 必须切换到 v1.7.0 版本 (新版本可能存在算子不兼容问题)
git checkout v1.7.0
```

#### Step 5: 推理 (Inference)

使用 predicate.py 进行推理生成

参数配置:

* **yaml_path:** 训练使用的 YAML 文件路径
* **ckpt_path:** 训练好的模型路径 (mindspore_model_final.ckpt 或微调后的模型)
* **TOKENIZER_DIR:** Spark-TTS-0.5B 分词器目录路径

```bash
python predicate.py
```

#### Step 6: 音频解码 (Decoding)

使用 decodec.py 运行 Bicodec 模型，将推理得到的 Token 解码为最终音频

```bash
python decodec.py
```

## Directory Structure

| Name            | Description              |
|:----------------|:-------------------------|
| `predata.py`    | 数据转换脚本 (Step 1)          |
| `datatomind.py` | MindRecord 生成脚本 (Step 2) |
| `convert.py`    | 模型权重转换脚本 (Step 3)        |
| `predicate.py`  | 音频解码脚本 (Step 6)          |
| `decodec.py`    | 项目依赖                     |

