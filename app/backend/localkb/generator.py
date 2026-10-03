"""生成封装：经 llama-server（OpenAI 兼容流式接口）产出带引用的回答。

模型档位与进程管理见 config.PROFILES 与 llm_server.py。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from . import config, llm_server


def _build_prompt(question: str, context: str) -> str:
    """拼装资料片段 + 用户问题（system 提示词由 llama-server 消息列表承载）。"""
    return (
        f"参考资料：\n{context}\n\n"
        f"问题：{question}\n"
        "请依据参考资料回答，并用 [编号] 标注引用。"
    )


def stream_answer(question: str, context: str,
                  profile: str = config.DEFAULT_PROFILE) -> Iterator[str]:
    """流式生成：逐段 yield 增量文本。"""
    llm_server.ensure_server(profile)
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, context)},
    ]
    resp = llm_server.chat_stream(
        messages,
        max_tokens=config.GEN_MAX_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        repeat_penalty=config.GEN_REPEAT_PENALTY,
    )
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            token = delta.get("content")
            if token:
                yield token
