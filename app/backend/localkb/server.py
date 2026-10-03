"""LocalKB FastAPI 服务（第 2 步）。

与 CLI 版的关键差异：FastAPI 与 llama-server 都常驻——嵌入模型进程内只加载
一次，llama-server 由服务生命周期管理，query 只发 HTTP，无 33s 冷启动税。

安全边界：仅绑定 127.0.0.1:8000（"数据不出本机"的硬约束）。

启动：
  cd E:\\LocalKB\\app\\backend
  .venv\\Scripts\\python -m uvicorn localkb.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config, llm_server, store
from .chunker import chunk_text
from .embedder import embed_texts, embed_query, get_model
from .retrieve import retrieve as do_retrieve

# ---------- 可变运行时设置（PUT /settings 修改） ----------
RUNTIME: dict[str, Any] = {
    "profile": config.DEFAULT_PROFILE,   # 生成档位
    "final_top_k": config.FINAL_TOP_K,   # 融合后保留条数
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动：预热嵌入模型 + 常驻 llama-server；关闭：显式回收。"""
    t0 = time.perf_counter()
    get_model()                              # BGE-M3 进程内单例，只此一次
    llm_server.ensure_server(RUNTIME["profile"])
    print(f"[LocalKB] 就绪：嵌入模型 + llama-server 已常驻 "
          f"（预热 {time.perf_counter() - t0:.1f}s）", flush=True)
    yield
    llm_server.stop_server()                 # Windows 子进程不随父死，必须显式收
    print("[LocalKB] 已关闭 llama-server", flush=True)


app = FastAPI(title="LocalKB API", version="0.2.0", lifespan=lifespan)


# ---------- 请求模型 ----------

class IngestBody(BaseModel):
    """二选一：给本地 path（服务端可访问），或直接给 text + source。"""
    path: str | None = None
    text: str | None = None
    source: str | None = None


class QueryBody(BaseModel):
    question: str
    profile: str | None = None      # 缺省用 RUNTIME["profile"]
    top_k: int | None = None        # 缺省用 RUNTIME["final_top_k"]


class SettingsBody(BaseModel):
    profile: str | None = None
    final_top_k: int | None = None


# ---------- 工具 ----------

def _ingest_chunks(chunks, source_label: str) -> dict[str, Any]:
    vectors = embed_texts([c.text for c in chunks])
    n = store.upsert_chunks(chunks, vectors)
    store.ensure_fts_index()
    return {"source": source_label, "chunks": n}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------- 接口 ----------

@app.get("/health")
def health() -> dict[str, Any]:
    stats = store.table_stats()
    return {
        "status": "ok",
        "embed_ready": True,          # lifespan 预热成功才会到达此处
        "llm_ready": llm_server.health_check(),
        "profile": RUNTIME["profile"],
        "index": {"rows": stats["rows"], "sources": stats["sources"]},
    }


@app.post("/ingest")
def ingest(body: IngestBody) -> dict[str, Any]:
    if body.text is not None:
        if not body.text.strip():
            raise HTTPException(400, "text 为空")
        source = body.source or f"inline-{int(time.time())}.txt"
        chunks = chunk_text(body.text.strip(), source)
        return _ingest_chunks(chunks, source)
    if body.path:
        p = Path(body.path)
        if not p.exists():
            raise HTTPException(404, f"路径不存在: {p}")
        from .chunker import ingest_path
        chunks = ingest_path(p)
        if not chunks:
            raise HTTPException(400, f"路径下没有可导入的 txt/md: {p}")
        return _ingest_chunks(chunks, p.name)
    raise HTTPException(400, "需要 path 或 (text+source) 之一")


@app.post("/query")
def query(body: QueryBody) -> StreamingResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question 为空")
    profile = body.profile or RUNTIME["profile"]
    if profile not in config.PROFILES:
        raise HTTPException(400, f"未知档位: {profile}，可选 {list(config.PROFILES)}")

    def gen() -> Iterator[str]:
        t0 = time.perf_counter()
        result = do_retrieve(question, top_k=body.top_k or RUNTIME["final_top_k"])
        yield _sse("meta", {
            "query": question,
            "hits": [{"cite_no": h["cite_no"], "source": h["source"],
                      "rrf_score": round(h["rrf_score"], 4)} for h in result["hits"]],
            "dense_count": result["dense_count"],
            "fts_count": result["fts_count"],
            "retrieval_elapsed": round(time.perf_counter() - t0, 2),
        })
        if not result["hits"]:
            yield _sse("done", {"reason": "no_hits"})
            return
        for token in llm_stream(profile, question, result["context"]):
            yield _sse("token", {"content": token})
        yield _sse("done", {"profile": profile,
                            "total_elapsed": round(time.perf_counter() - t0, 2)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


def llm_stream(profile: str, question: str, context: str) -> Iterator[str]:
    from . import generator
    yield from generator.stream_answer(question, context, profile=profile)


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    return {"profile": RUNTIME["profile"],
            "final_top_k": RUNTIME["final_top_k"],
            "available_profiles": list(config.PROFILES)}


@app.put("/settings")
def put_settings(body: SettingsBody) -> dict[str, Any]:
    if body.profile is not None:
        if body.profile not in config.PROFILES:
            raise HTTPException(400, f"未知档位: {body.profile}")
        RUNTIME["profile"] = body.profile
        llm_server.ensure_server(body.profile)   # 档位变化时自动重启 llama-server
    if body.final_top_k is not None:
        if not 1 <= body.final_top_k <= 20:
            raise HTTPException(400, "final_top_k 取值 1~20")
        RUNTIME["final_top_k"] = body.final_top_k
    return get_settings()
