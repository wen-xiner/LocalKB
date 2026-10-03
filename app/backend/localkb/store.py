"""LanceDB 向量库封装：建表、写入、dense 检索与 FTS(BM25) 检索。"""

from __future__ import annotations

from typing import Any

from . import config
from .chunker import Chunk

TABLE_NAME = "kb"


def get_db() -> Any:
    """打开（必要时创建）索引目录。"""
    import lancedb  # 懒加载：stats 等轻命令不强制要求依赖就绪

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(config.INDEX_DIR))


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """把块与向量写入（追加模式），返回写入条数。"""
    if len(chunks) != len(vectors):
        raise ValueError(f"块数 {len(chunks)} 与向量数 {len(vectors)} 不一致")
    rows = [
        {
            "id": c.id,
            "vector": v,
            "text": c.text,
            "source": c.source,
            "chunk_index": c.chunk_index,
        }
        for c, v in zip(chunks, vectors)
    ]
    db = get_db()
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        tbl.add(rows)
    else:
        db.create_table(TABLE_NAME, data=rows, mode="create")
    return len(rows)


def ensure_fts_index() -> None:
    """为 vector / text 列建索引（向量 IVF + BM25 全文）。已存在则跳过。"""
    from lancedb.index import FTS, IvfFlat

    db = get_db()
    if TABLE_NAME not in db.table_names():
        raise FileNotFoundError("索引表不存在，请先执行 ingest")
    tbl = db.open_table(TABLE_NAME)
    try:
        tbl.create_index("vector", config=IvfFlat(distance_type="l2"), replace=False)
    except Exception:
        pass  # 已存在
    try:
        # 中文暂用 icu/split（≈单字切），效果不足时换 jieba 预分词
        tbl.create_index(
            "text",
            config=FTS(base_tokenizer="icu/split", stem=False,
                       remove_stop_words=False, lower_case=True),
            replace=False,
        )
    except Exception:
        pass  # 已存在


def table_stats() -> dict[str, Any]:
    """返回索引规模统计。"""
    db = get_db()
    if TABLE_NAME not in db.table_names():
        return {"exists": False, "rows": 0, "sources": []}
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_arrow().select(["source"]).to_pandas()
    sources = sorted(df["source"].unique().tolist())
    return {"exists": True, "rows": tbl.count_rows(), "sources": sources}


def dense_search(query_vec: list[float], top_k: int) -> list[dict[str, Any]]:
    """向量检索 top-k。"""
    db = get_db()
    tbl = db.open_table(TABLE_NAME)
    rows = (
        tbl.search(query_vec)
        .limit(top_k)
        .to_list()
    )
    return [{"id": r["id"], "text": r["text"], "source": r["source"],
             "score": float(r.get("_distance", 0.0))} for r in rows]


def fts_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """BM25 全文检索 top-k。"""
    db = get_db()
    tbl = db.open_table(TABLE_NAME)
    try:
        rows = tbl.search(query, query_type="fts").limit(top_k).to_list()
    except Exception:
        # FTS 索引尚未建立或查询无法分词时静默降级
        return []
    return [{"id": r["id"], "text": r["text"], "source": r["source"],
             "score": float(r.get("score", 0.0))} for r in rows]
