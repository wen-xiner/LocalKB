# LocalKB 后端（第 1 步：命令行链路）

目标：不碰界面，纯 Python 命令行跑通「导入 → 切分 → 嵌入 → 检索 → 生成」。

## 目录

```
localkb/
  config.py      路径、档位（含本机 6GB 显存的档位判断）、检索参数
  chunker.py     导入 + 结构感知切分（md 标题优先，段落兜底，滑窗长段）
  embedder.py    BGE-M3 dense 向量（懒加载，cuda fp16）
  store.py       LanceDB：建表 / 写入 / dense 检索 / FTS(BM25) 检索
  retrieve.py    双路召回 + RRF 融合，产出带 [编号] 的引用片段
  llm_server.py  llama-server 子进程管理（启动 / 健康检查 / 停止）
  generator.py   经 llama-server 流式生成带引用回答
  cli.py         ingest / ask / stats 三个子命令
tests/           纯单测（不依赖 torch 等重型库）
```

## 生成架构说明（重要）

生成不走 Python 库绑定（llama-cpp-python 在 PyPI 无 Windows 预编译 wheel，
安装需本地编译），改用 **llama.cpp 官方预编译 CUDA 版 `llama-server.exe`**：
ask 命令会自动以子进程方式拉起 llama-server（仅绑定 127.0.0.1:8355），
Python 侧经 OpenAI 兼容 `/v1/chat/completions` 流式调用。该形态与最终
Tauri sidecar 打包架构一致，日志落在 `data/logs/`。

## 环境准备

```powershell
cd E:\LocalKB\app\backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 模型准备

**本机网络限制**：hf-mirror / GitHub 资产 / 多数 CDN 直连均不通或不稳，
实测唯一快通道为 **魔搭 ModelScope 直连**（4~6MB/s）与清华/腾讯 PyPI 镜像。
以下均按已验证可用的渠道执行（2026-10-02 实测，两个模型已就位）：

1. **嵌入模型 BGE-M3**（2.27GB）→ `E:\LocalKB\models\embedding\bge-m3\`
   魔搭 `BAAI/bge-m3` 仓库全套（pytorch_model.bin + tokenizer + config）
2. **生成模型 Qwen3-4B Q5_K_M**（2.89GB）→ `E:\LocalKB\models\qwen3\qwen3-4b-q5_k_m.gguf`
   魔搭 `Qwen/Qwen3-4B-GGUF` 的 `Qwen3-4B-Q5_K_M.gguf`
3. **llama-server**（CUDA 12.4 x64）→ `E:\LocalKB\models\llama-server\`
   GitHub `ggml-org/llama.cpp` release `b11335` 的
   `llama-b11335-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip`，
   解压后需保证 `llama-server.exe` 与 cudart 的 dll 同目录。

档位说明（本机 RTX 4050 Laptop 6GB 显存实测判断）：
- `default` = Qwen3-4B Q5_K_M 全显存（~2.9GB 权重 + KV 余量，最稳）
- `high`    = Qwen3-8B Q4_K_M（~5GB，6GB 卡放不下全量，默认 2/3 层进 GPU 其余走 CPU）
- `cloud`   = Qwen3-14B（云 GPU 租用场景，本机不落盘）

## 运行

```powershell
# 放几个 txt/md 到 data\corpus\ 后：
.venv\Scripts\python -m localkb.cli ingest E:\LocalKB\data\corpus
.venv\Scripts\python -m localkb.cli stats
.venv\Scripts\python -m localkb.cli ask "什么是混合检索"
.venv\Scripts\python -m localkb.cli ask "..." --profile high
```

判定标准：ingest 显示写入块数无报错；ask 先打印双路召回与引用列表，再流式输出
带 `[编号]` 引用的回答；断网状态下 ask 仍可完整出答案（离线闭环证据）。

## 已知边界（第 1 步有意不做）

- 仅支持 txt / md；pdf / docx 解析放第 2 步
- sparse 词法权重融合（BGE-M3 特色）留了接口，先用 dense+BM25+RRF 跑通
- 中文 FTS 分词是按单字兜底，效果不足时再换 jieba 预分词
