#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MultiHop-RAG 环境自检脚本

用法：在项目根目录（含 dataset/ models/ code/ 的目录）执行
    python scripts/check_env.py

检查项：Python 版本 / 关键依赖 / 本地模型 / 基准数据 / 推理服务（llama-server）。
只依赖标准库 + 被测包本身；每项独立捕获异常，输出体检报告。
"""
import importlib.util
import json
import os
import sys
import urllib.request

ROOT = os.getcwd()
RESULTS = []


def record(name, run):
    try:
        ok, detail = run()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"异常: {type(exc).__name__}: {exc}"
    RESULTS.append((name, ok, detail))


# ---------- 1. Python 版本 ----------
record(
    "Python 3.11",
    lambda: (sys.version_info[:2] == (3, 11), sys.version.split()[0]),
)

# ---------- 2. 关键依赖 ----------
PACKAGES = [
    ("llama_index", "llama-index==0.9.40"),
    ("sentence_transformers", "sentence-transformers"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("rank_bm25", "rank-bm25"),
    ("openai", "openai"),
    ("tqdm", "tqdm"),
    ("dotenv", "python-dotenv"),
    ("tiktoken", "tiktoken"),
    ("nltk", "nltk"),
]
for module, label in PACKAGES:
    record(
        f"依赖 {label}",
        (lambda m=module: (importlib.util.find_spec(m) is not None, "可导入" if importlib.util.find_spec(m) else "缺失，pip install 补装")),
    )


def check_torch():
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return True, f"{torch.__version__} | CUDA: {torch.cuda.get_device_name(0)}"
    return True, f"{torch.__version__} | CPU 模式（可用；重排较慢）"


record("PyTorch（含 GPU 探测）", check_torch)


# ---------- 3. 基准数据 ----------
def check_corpus():
    path = os.path.join(ROOT, "dataset", "corpus.json")
    if not os.path.isfile(path):
        return False, "缺失 dataset/corpus.json（MultiHop-RAG 官方语料，609 篇）"
    data = json.load(open(path, encoding="utf-8"))
    return len(data) == 609, f"{len(data)} 篇（期望 609）"


def check_queries():
    path = os.path.join(ROOT, "dataset", "MultiHopRAG.json")
    if not os.path.isfile(path):
        return False, "缺失 dataset/MultiHopRAG.json（官方题库，2556 题）"
    data = json.load(open(path, encoding="utf-8"))
    return len(data) == 2556, f"{len(data)} 题（期望 2556）"


record("数据 corpus.json", check_corpus)
record("数据 MultiHopRAG.json", check_queries)


# ---------- 4. 本地模型 ----------
def dir_ok(path):
    return os.path.isdir(path) and any(os.scandir(path))


record(
    "模型 all-MiniLM-L6-v2",
    lambda: (
        dir_ok(os.path.join(ROOT, "models", "all-MiniLM-L6-v2")),
        "models/all-MiniLM-L6-v2 就绪" if dir_ok(os.path.join(ROOT, "models", "all-MiniLM-L6-v2")) else "缺失（见 SKILL.md 步骤 2）",
    ),
)
record(
    "模型 bge-reranker-v2-m3",
    lambda: (
        dir_ok(os.path.join(ROOT, "models", "bge-reranker-v2-m3")),
        "models/bge-reranker-v2-m3 就绪" if dir_ok(os.path.join(ROOT, "models", "bge-reranker-v2-m3")) else "缺失（见 SKILL.md 步骤 2）",
    ),
)


# ---------- 5. 推理服务 ----------
def check_llm():
    base = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:18080/v1")
    health_url = base.replace("/v1", "") + "/health"
    with urllib.request.urlopen(health_url, timeout=5) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", "replace")
    ok = '"ok"' in body and resp.status == 200
    return ok, f"{health_url} -> {body[:60]}"


record("推理服务 llama-server", check_llm)


# ---------- 输出 ----------
def main():
    print("=" * 64)
    print("MultiHop-RAG 环境自检报告")
    print("=" * 64)
    failed = 0
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name:<28} {detail}")
    print("-" * 64)
    if failed == 0:
        print("全部通过 —— 环境配置完成，可以开始冒烟测试（SKILL.md 步骤 4）。")
    else:
        print(f"{failed} 项未通过 —— 按 SKILL.md 与 references/pitfalls.md 排查后重跑本脚本。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
