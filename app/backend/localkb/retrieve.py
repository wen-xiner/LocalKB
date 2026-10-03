"""检索编排：dense + BM25 双路召回，RRF 融合，供生成阶段引用。"""

from __future__ import annotations

from typing import Any

from . import config, embedder, store


def _rrf_fuse(dense_hits: list[dict], fts_hits: list[dict],
              k: int = config.RRF_K, final_k: int = config.FINAL_TOP_K) -> list[dict]:
    """Reciprocal Rank Fusion：按 id 去重合并两路排名。"""
    scores: dict[str, float] = {}
    hits_by_id: dict[str, dict] = {}
    for rank, h in enumerate(dense_hits, start=1):
        scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (k + rank)
        hits_by_id.setdefault(h["id"], h)
    for rank, h in enumerate(fts_hits, start=1):
        scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (k + rank)
        hits_by_id.setdefault(h["id"], h)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for i, (cid, score) in enumerate(ranked[:final_k], start=1):
        h = dict(hits_by_id[cid])
        h["cite_no"] = i
        h["rrf_score"] = score
        out.append(h)
    return out


def retrieve(query: str, top_k: int | None = None) -> dict[str, Any]:
    """完整检索：返回融合后的片段列表与引用文本块。top_k 覆盖 config.FINAL_TOP_K。"""
    final_k = top_k or config.FINAL_TOP_K
    qvec = embedder.embed_query(query)
    dense_hits = store.dense_search(qvec, config.RETRIEVAL_TOP_K)
    fts_hits = store.fts_search(query, config.RETRIEVAL_TOP_K)
    fused = _rrf_fuse(dense_hits, fts_hits, final_k=final_k)

    # 引用文本：[1] 来源 :: 片段正文
    context_blocks = [
        f"[{h['cite_no']}] ({h['source']}) {h['text']}" for h in fused
    ]
    return {
        "query": query,
        "hits": fused,
        "dense_count": len(dense_hits),
        "fts_count": len(fts_hits),
        "context": "\n\n".join(context_blocks),
    }
