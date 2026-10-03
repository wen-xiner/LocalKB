"""llama-server 子进程管理：启动本地 OpenAI 兼容服务并做健康检查。

为什么用 llama-server 而不是 llama-cpp-python：
  llama-cpp-python 在 PyPI 上只有源码包，Windows 安装需本地编译
  （CMake + MSVC + CUDA Toolkit）；llama.cpp 官方提供预编译 CUDA 二进制，
  llama-server 即其内置的 OpenAI 兼容服务，与最终 Tauri sidecar 架构一致。

只绑定 127.0.0.1，日志重定向到文件（不用 PIPE，避免缓冲区挂死）。
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config

_process: subprocess.Popen | None = None
_log_file: Path | None = None
_loaded_profile: str | None = None


def server_exe() -> Path:
    """llama-server.exe 路径（models/llama-server/ 解压后）。"""
    return config.MODELS_DIR / "llama-server" / "llama-server.exe"


def _pick_args(profile: str) -> list[str]:
    cfg = config.PROFILES[profile]
    gguf = config.MODELS_QWEN_DIR / cfg["gguf"]
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF 不存在: {gguf}（见 README 模型准备）")
    return [
        "-m", str(gguf),
        "-ngl", str(cfg["n_gpu_layers"]),
        "-c", str(cfg["n_ctx"]),
        "--host", "127.0.0.1",
        "--port", str(config.SERVER_PORT),
    ]


def ensure_server(profile: str = config.DEFAULT_PROFILE) -> None:
    """确保 llama-server 已启动且模型加载完成（幂等；档位变化时自动重启）。"""
    global _process, _log_file, _loaded_profile
    if (_process is not None and _process.poll() is None
            and _loaded_profile == profile and health_check()):
        return
    if _process is not None and _process.poll() is None:
        stop_server()   # 档位变了：先收旧实例再起新的

    exe = server_exe()
    if not exe.exists():
        raise FileNotFoundError(
            f"未找到 {exe}\n请解压 llama.cpp CUDA 包到 models/llama-server/（见 README）"
        )
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = config.LOG_DIR / f"llama-server-{profile}.log"
    log_handle = open(_log_file, "ab")
    _process = subprocess.Popen(
        [str(exe), *_pick_args(profile)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    # 等待模型加载：健康检查轮询，超时 120s（4B Q5 加载约 10~30s）
    deadline = time.time() + 120
    while time.time() < deadline:
        if _process.poll() is not None:
            tail = _log_tail()
            raise RuntimeError(f"llama-server 启动失败，日志尾部：\n{tail}")
        if health_check():
            _loaded_profile = profile
            return
        time.sleep(1.0)
    raise TimeoutError(f"llama-server 健康检查超时，日志：{_log_file}")


def health_check() -> bool:
    """GET /health：200 = 就绪。"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{config.SERVER_PORT}/health", timeout=2
        ) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _log_tail(lines: int = 15) -> str:
    if _log_file is None or not _log_file.exists():
        return "(无日志)"
    content = _log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def chat_stream(messages: list[dict[str, str]],
                max_tokens: int, temperature: float, top_p: float,
                repeat_penalty: float) -> "json._JsonLoader":
    """调用 /v1/chat/completions 流式接口，返回逐行 SSE 的响应对象。"""
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{config.SERVER_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=300)


def stop_server() -> None:
    """停止 llama-server 子进程。"""
    global _process, _loaded_profile
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None
    _loaded_profile = None
