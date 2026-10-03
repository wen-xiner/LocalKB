"""切分逻辑单测：不依赖任何重型库（无 torch / lancedb / llama.cpp），可直接跑。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localkb.chunker import chunk_text  # noqa: E402


def test_md_heading_split():
    text = "# 标题一\n内容A第一段。\n\n内容A第二段。\n\n## 标题二\n内容B。"
    chunks = chunk_text(text, "demo.md")
    assert len(chunks) >= 2
    assert any("标题一" in c.text for c in chunks)
    assert any("标题二" in c.text for c in chunks)


def test_plain_paragraph_split():
    text = "段落一内容。\n\n段落二内容。\n\n段落三内容。"
    chunks = chunk_text(text, "plain.txt")
    assert all("段落" in c.text for c in chunks)
    assert chunks[0].source == "plain.txt"


def test_chunk_ids_sequential():
    text = "# A\n" + "长" * 300 + "\n\n# B\n短内容"
    chunks = chunk_text(text, "x.md")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.id.startswith("x.md::") for c in chunks)


def test_long_section_sliding_window():
    text = "字" * 1500
    chunks = chunk_text(text, "long.txt", size=500, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 500 for c in chunks)


def test_empty_and_none():
    assert chunk_text("", "empty.txt") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all chunker tests passed")
