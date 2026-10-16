---
name: multihop-rag-setup
description: "为 MultiHop-RAG 检索增强问答项目（github.com/lcw578/RAG_project）在 Windows 机器上从零安装配置完整运行环境：conda Python 环境、本地检索/重排模型、llama.cpp + Qwen3.5-9B 推理服务、MultiHop-RAG 数据，并执行端到端冒烟验证。This skill should be used when onboarding a new member to the project, setting up its environment on a fresh machine, or when the user asks how to install, configure or reproduce this RAG project environment. 触发词：安装环境、配置环境、新机器部署、复现 RAG 项目、environment setup。"
agent_created: true
---

# MultiHop-RAG 环境配置向导

## 用途

把 `github.com/lcw578/RAG_project` 的完整运行环境装到一台 Windows 机器上，并验证到「冒烟测试跑通出分」。
环境由四部分组成：① conda Python 环境 ② 两个本地检索模型 ③ 一个 llama.cpp 推理服务（Qwen3.5-9B）④ 基准数据。

## 前置条件（先逐项核对）

| 项 | 要求 | 检查命令 |
|---|---|---|
| 系统 | Windows 10 / 11 | — |
| conda | 任意发行 | `conda --version` |
| Git | 任意（推送需先配凭据） | `git --version` |
| 磁盘 | ≥ 30 GB 空闲（模型 ~12GB + 环境 + 数据） | — |
| 网络 | 可访问 hf-mirror；GitHub 可达（大文件建议挂代理） | — |
| GPU（可选） | 有 NVIDIA GPU 则重排加速；纯 CPU 亦可运行（慢 5-10 倍） | `nvidia-smi` |
| 内存 | ≥ 16 GB；**同一脚本绝不重复加载大模型**（本项目曾因双份加载卡死） | — |

## 配置流程（按序执行，每步有验证）

### 步骤 0：获取代码与数据

```bash
git clone https://github.com/lcw578/RAG_project.git
cd RAG_project && mkdir -p dataset output qa_output logs

# 数据：MultiHop-RAG 官方仓库 github.com/yixuantt/MultiHop-RAG
#   取 corpus.json（609 篇语料）与 MultiHopRAG.json（2556 题）放入 dataset/
```

**验证**：`dataset/` 下有 `corpus.json` 与 `MultiHopRAG.json`。

### 步骤 1：Python 环境

```bash
conda create -n mhr python=3.11 -y
conda activate mhr
pip install -r requirements.txt

# CUDA 版 PyTorch（有 NVIDIA GPU 时安装；国内用 SJTU 镜像——清华 pytorch-wheels 已失效）
pip install torch==2.14.1+cu126 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
```

**验证**：运行本 skill 的自检脚本 `python scripts/check_env.py`（逐项检查依赖/模型/数据/服务）。

### 步骤 2：本地检索模型（放入 `models/`）

```bash
# hf CLI 新版默认 Xet 协议，与镜像不兼容——必须禁用（否则 401）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3 --exclude "onnx/*"
```

### 步骤 3：LLM 推理服务（llama.cpp + Qwen3.5-9B）

```bash
# 模型：Qwen3.5-9B Q8_0 GGUF（约 9.8GB；bartowski 量化，ModelScope / hf-mirror 可取）
# llama.cpp：取 Windows CUDA 预编译版（或自行编译），启动 OpenAI 兼容服务：
llama-server -m <模型目录>/Qwen_Qwen3.5-9B-Q8_0.gguf --port 18080 -ngl 99 -c 8192 --jinja
```

要点：
- 端口**不要用 8080**（常被代理软件占用）——本项目统一 **18080**，客户端的 `LOCAL_LLM_BASE_URL` 环境变量可覆盖
- 显存建议 ≥ 11GB（Q8_0 约 9.8GB）：不足时改用更小量化（如 Q4_K_M ≈ 5.8GB，精度略降）
- 客户端两个必须：思考模式由代码通过 `chat_template_kwargs` 关闭；HTTP 每题新建连接（见坑位表）

### 步骤 4：冒烟测试（20 题小样本，端到端）

```bash
# ① 混合检索（首次运行会构建索引，约 2 分钟）
python code/retrieval/run_hybrid_retrieval.py --queries dataset/MultiHopRAG.json \
  --start 0 --limit 20 --retrieve_top_k 10 --rrf_top_k 10 --rerank_top_n 10 \
  --output output/smoke_query_result.json

# ② 生成（prompt v3）
python code/generation/qa_llama_v3.py --input output/smoke_query_result.json --output qa_output/smoke.json

# ③ 官方评测
python code/evaluation/qa_evaluate.py --file qa_output/smoke.json
```

**预期**：全程无报错，评测出分（20 题小样本总体约 55-80%，波动正常）→ 环境配置完成。

## 完整复现（可选）

全量 2556 题的六层增强流程见项目 README「复现步骤」；分块断点续跑执行器：`code/runners/run_qa_full.sh`。

## 坑位速查（详细版见 references/pitfalls.md）

| 症状 | 原因 | 解决 |
|---|---|---|
| `No module named 'pkg_resources'` | setuptools ≥81 已移除该模块（llama-index 0.9.x 依赖） | `pip install "setuptools<81"` |
| hf download 报 **401** | 新版 hf 默认 Xet 协议与 hf-mirror 不兼容 | `HF_HUB_DISABLE_XET=1` |
| torch 装不上（清华源 404） | 清华 pytorch-wheels 镜像已下线 | 换 SJTU 镜像（见步骤 1） |
| llama-server 启动失败 | 8080 端口被占用 | 改 18080 |
| 模型只输出思维链、无答案 | Qwen3.5 默认开启思考模式 | 客户端传 `chat_template_kwargs: {"enable_thinking": false}` |
| 复用连接调用「隔次 404」 | llama.cpp keep-alive 兼容性缺陷（严格交替复现） | 每题新建 HTTP 连接（`qa_llama*.py` 已内置 `[ADAPTED]` 修复） |
| 整机卡死 | 同时加载多份 2.3GB 模型 + 内存不足 | 一次只加载一份；大模型加载前先查可用内存 |
| 中文路径下脚本异常 | 工具链编码兼容性 | 模型/项目路径尽量用 ASCII |

## 资源

- `scripts/check_env.py` — 环境自检脚本：依赖 / 模型 / 数据 / 推理服务四项全查，输出体检报告
- `references/setup-details.md` — 详细命令参考（镜像、模型下载、服务部署、可选组件）
- `references/pitfalls.md` — 坑位手册（症状 → 原因 → 解决，附实测依据与验证方法）
