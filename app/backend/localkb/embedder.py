"""BGE-M3 嵌入封装：dense 向量（1024 维）+ FTS 关键词，懒加载单例。

sparse 词法权重（BGE-M3 特色）先留接口，第 1 步用 dense + LanceDB FTS(BM25)
跑通链路；sparse 融合作为后续增强，不阻塞主线。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import config


@lru_cache(maxsize=1)
def get_model() -> Any:
    """加载 BGE-M3（首次调用会占用数秒；需先在 config.EMBED_MODEL_DIR 放好权重）。"""
    if not config.EMBED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"嵌入模型目录不存在: {config.EMBED_MODEL_DIR}\n"
            "请下载 BGE-M3 到该目录（见 README 模型准备一节）"
        )
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        str(config.EMBED_MODEL_DIR),
        use_fp16=True,   # 6GB 显存跑 0.6B 模型 fp16 无压力
        devices=["cuda:0"],
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量产出 dense 向量（1024 维 list[float]）。"""
    if not texts:
        return []
    model = get_model()
    out = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    return [row.tolist() for row in dense]


def embed_query(query: str) -> list[float]:
    """单条查询向量。"""
    return embed_texts([query])[0]
