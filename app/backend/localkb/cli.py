"""命令行入口：ingest / ask / stats 三个子命令。

用法示例：
  python -m localkb.cli ingest E:\\LocalKB\\data\\corpus
  python -m localkb.cli stats
  python -m localkb.cli ask "什么是混合检索"
  python -m localkb.cli ask "..." --profile high
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config


def cmd_ingest(args: argparse.Namespace) -> int:
    from . import chunker, embedder, store  # 懒加载重依赖

    path = Path(args.path).resolve()
    t0 = time.perf_counter()
    chunks = chunker.ingest_path(path)
    print(f"[1/3] 导入并切分 {len(chunks)} 块，开始嵌入…")
    vectors = embedder.embed_texts([c.text for c in chunks])
    print(f"[2/3] 嵌入完成，写入 LanceDB 索引 {config.INDEX_DIR} …")
    n = store.upsert_chunks(chunks, vectors)
    store.ensure_fts_index()
    dt = time.perf_counter() - t0
    print(f"[3/3] 完成：写入 {n} 块，耗时 {dt:.1f}s")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from . import generator, retrieve  # 懒加载重依赖

    t0 = time.perf_counter()
    result = retrieve.retrieve(args.question)
    dt = time.perf_counter() - t0
    print(f"--- 检索（dense {result['dense_count']} 路 + FTS {result['fts_count']} 路，"
          f"融合 {len(result['hits'])} 条，耗时 {dt:.2f}s）---")
    if not result["hits"]:
        print("知识库中没有检索到相关内容，请先执行 ingest 导入文档。")
        return 1
    for h in result["hits"]:
        print(f"  [{h['cite_no']}] {h['source']} (rrf={h['rrf_score']:.4f})")
    print("--- 回答（流式）---")
    try:
        for token in generator.stream_answer(args.question, result["context"],
                                             profile=args.profile):
            print(token, end="", flush=True)
        print()
    finally:
        # Windows 下 Popen 子进程不随父进程退出，必须显式停服，
        # 否则 llama-server 残留占 ~3GB 显存，拖慢后续所有 GPU 任务
        from . import llm_server
        llm_server.stop_server()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from . import store  # 懒加载重依赖

    s = store.table_stats()
    if not s["exists"]:
        print("索引尚未建立（data/index 下无 kb 表）。先执行 ingest。")
        return 0
    print(f"索引: {config.INDEX_DIR}")
    print(f"块数: {s['rows']}")
    print(f"来源文档({len(s['sources'])}): " + ", ".join(s["sources"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localkb",
                                     description="LocalKB 命令行（第 1 步：纯 Python 跑通链路）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="导入目录/文件：切分→嵌入→入库")
    p_ing.add_argument("path", help="文档目录或单个 txt/md 文件")
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="提问：检索→融合→流式生成")
    p_ask.add_argument("question", help="问题文本")
    p_ask.add_argument("--profile", default=config.DEFAULT_PROFILE,
                       choices=list(config.PROFILES), help="生成模型档位")
    p_ask.set_defaults(func=cmd_ask)

    p_st = sub.add_parser("stats", help="查看索引统计")
    p_st.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"[缺文件] {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"[缺依赖] {e}\n请先安装依赖: pip install -r requirements.txt", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
