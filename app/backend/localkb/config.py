"""全局配置：路径、模型档位、检索参数。

路径布局（与工作区根对齐）：
  E:\\LocalKB\\
    data\\corpus\\    原始文档（txt / md）
    data\\index\\     LanceDB 索引
    models\\
      embedding\\bge-m3\\     BGE-M3 本地目录
      qwen3\\                 GGUF 权重文件
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------- 离线声明（必须在 transformers / FlagEmbedding 导入前生效） ----------
# 断网实测：transformers 默认会尝试连 HF Hub 检查模型版本，离线时白等约 80s
# 超时才回退本地。本产品全程离线，直接声明离线模式跳过联网检查。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[3]      # E:\LocalKB（localkb → backend → app → 根）
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"                # 待导入的原始文档
INDEX_DIR = DATA_DIR / "index"                  # LanceDB 库目录

MODELS_DIR = ROOT / "models"
EMBED_MODEL_DIR = MODELS_DIR / "embedding" / "bge-m3"
MODELS_QWEN_DIR = MODELS_DIR / "qwen3"

# ---------- llama-server（本地 OpenAI 兼容推理服务） ----------
SERVER_PORT = 8355            # 仅绑定 127.0.0.1
LOG_DIR = ROOT / "data" / "logs"

# ---------- 嵌入模型（BGE-M3，dense 1024 维 + sparse 词法权重） ----------
EMBED_BATCH_SIZE = 8

# ---------- 切分参数（中文场景按字符估算，1 token ≈ 1~1.5 字） ----------
CHUNK_SIZE = 500        # 单块目标字符数
CHUNK_OVERLAP = 80      # 相邻块重叠字符数

# ---------- 检索参数 ----------
RETRIEVAL_TOP_K = 5     # 每路召回条数（dense / FTS 各取 top-k）
FINAL_TOP_K = 5         # RRF 融合后保留条数
RRF_K = 60              # RRF 平滑常数

# ---------- 生成模型档位 ----------
# 本机实测（2026-10-02）：RTX 4050 Laptop 6GB 显存 + 16GB 内存 + Win11 24H2。
# 8B Q4_K_M 权重约 5GB，6GB 卡扣掉系统占用后放不下权重 + KV cache，
# 故默认档用 4B 全量进显存；8B 作为"高配档"默认走部分层 offload。
PROFILES: dict[str, dict] = {
    "light": {  # 纯 CPU，兜底
        "gguf": "qwen3-4b-q4_k_m.gguf",
        "n_gpu_layers": 0,
        "n_ctx": 4096,
    },
    "default": {  # 本机默认：4B Q5_K_M 全显存，留 KV 余量
        "gguf": "qwen3-4b-q5_k_m.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
    },
    "high": {  # 8B Q4_K_M：部分层进显存 + CPU offload（16GB 内存撑得住）
        "gguf": "qwen3-8b-q4_k_m.gguf",
        "n_gpu_layers": 20,   # 约 2/3 层进 GPU，其余走 CPU
        "n_ctx": 8192,
    },
    "cloud": {  # 云 GPU（租用）参考：14B Q4 全显存，本机不落盘
        "gguf": "qwen3-14b-q4_k_m.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 16384,
    },
}

DEFAULT_PROFILE = "default"

# ---------- 生成参数 ----------
GEN_MAX_TOKENS = 1024
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.8
GEN_REPEAT_PENALTY = 1.05

SYSTEM_PROMPT = (
    "你是一个私有化本地知识库助手。仅根据提供的参考资料回答问题，"
    "不得编造资料以外的内容。回答时用 [1] [2] 标注所引用资料的编号；"
    "若参考资料不足以回答，直接说明资料中未涉及，不要猜测。"
)
