"""导入与切分：读取 txt / md 文档，按标题与段落结构切分为固定大小的块。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_MD_HEADING = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)


@dataclass
class Chunk:
    """一个知识块：正文文本 + 来源信息。"""

    id: str
    text: str
    source: str        # 相对于 corpus 目录的文件名
    chunk_index: int


def read_document(path: Path) -> str:
    """读取单个文档为纯文本（当前支持 txt / md）。"""
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError(f"暂不支持的格式: {path.suffix}（{path.name}），当前仅 txt/md")
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.strip()


def _split_by_structure(text: str) -> list[str]:
    """优先按 Markdown 标题切；无标题则按空行段落切。"""
    sections: list[str] = []
    matches = list(_MD_HEADING.finditer(text))
    if matches:
        # 第一个标题之前的引言
        if text[: matches[0].start()].strip():
            sections.append(text[: matches[0].start()].strip())
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[m.start():end].strip())
    else:
        sections = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [s for s in sections if s]


def _pack(sections: list[str], size: int, overlap: int) -> list[str]:
    """把小节合并到目标大小；超长小节内部滑窗切分。"""
    chunks: list[str] = []
    buf = ""
    for sec in sections:
        if len(sec) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            step = max(size - overlap, 1)
            for i in range(0, len(sec), step):
                piece = sec[i:i + size]
                if len(piece) >= overlap or i == 0:
                    chunks.append(piece)
        elif len(buf) + len(sec) + 1 <= size or not buf:
            buf = f"{buf}\n{sec}".strip()
        else:
            chunks.append(buf)
            buf = sec
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(text: str, source: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """切分一段文本，返回带 id 与来源的 Chunk 列表。

    有 Markdown 标题时：每个标题节是独立语义单元，只做节内滑窗，不跨节合并；
    无标题时：段落做空间填充合并到目标大小。
    """
    sections = _split_by_structure(text)
    has_headings = bool(_MD_HEADING.search(text))
    if has_headings:
        packed: list[str] = []
        for sec in sections:
            packed.extend(_pack([sec], size, overlap))
    else:
        packed = _pack(sections, size, overlap)
    safe_src = re.sub(r"[^\w.-]+", "_", source)
    return [
        Chunk(id=f"{safe_src}::{i}", text=c, source=source, chunk_index=i)
        for i, c in enumerate(packed)
    ]


def ingest_path(path: Path) -> list[Chunk]:
    """导入单个文件（或目录下所有 txt/md），返回全部块。"""
    files = (
        sorted(p for p in path.iterdir() if p.suffix.lower() in {".txt", ".md"})
        if path.is_dir() else [path]
    )
    if not files:
        raise FileNotFoundError(f"没有找到可导入的 txt/md 文件: {path}")
    chunks: list[Chunk] = []
    for f in files:
        text = read_document(f)
        if text:
            src = f.name if path.is_dir() else f.name
            chunks.extend(chunk_text(text, src))
    return chunks
