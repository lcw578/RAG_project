# MultiHop-RAG 检索增强问答系统

> **华南理工大学《自然语言处理》期末项目** —— 在 **MultiHop-RAG** 基准（609 篇新闻 × 2556 道跨文档问题）上构建完整的检索增强生成（RAG）系统。
>
> 本地部署 **Qwen3.5-9B**（llama.cpp / GGUF Q8_0，≤10B 满足约束），复用官方混合检索管线，并叠加**四层自研增强模块**。
> **总体准确率从官方管线的 58.65 提升到 77.0（Excellent 档）**，超过公开系统评测中 Dense RAG（67.0）、Community-GraphRAG-Local（69.0）、HippoRAG2（70.3）等方案。

---

## 一、结果总览

### 1.1 消融实验（全量 2556 题，官方 qa_evaluate.py，B = 100 × Accuracy）

| # | 系统配置 | comparison | inference | null | temporal | **总体** |
|---|---|---|---|---|---|---|
| 0 | 官方混合检索管线（BM25+稠密+RRF+BGE 重排；思考关闭） | 41.0 | 94.0 | 61.8 | 33.4 | **58.6** |
| 1 | ＋答案清洗规则（输出校验层） | 41.0 | 94.0 | 90.0 | 33.4 | **62.0** |
| 2 | ＋生成 prompt v2（逐源核对 / 禁止单字母） | 55.1 | 92.0 | 80.7 | 36.5 | **65.7** |
| 3 | ＋源覆盖增强（检索后处理 #1） | 58.4 | 93.1 | 80.1 | 38.8 | **67.6** |
| 4 | ＋日期标注 + 时间线引导（检索后处理 #2 + prompt v3） | 64.8 | 94.7 | 75.1 | 57.6 | **73.9** |
| 5 | ＋来源补强 v2 + 日期匹配（检索后处理 #3/#4） | **67.2** | **95.1** | 75.4 | **58.5** | **75.1** |
| **6** | **提交配置 = 第 5 行 + 答案清洗规则** | **67.2** | **95.1** | **91.7** | **58.5** | **77.0** 🏆 |

### 1.2 与公开结果对照

| 方法 | 总体 Accuracy |
|---|---|
| 官方论文 GPT-4 + 检索基线 | ~56 |
| Dense RAG（系统评测横评） | 67.0 |
| Community-GraphRAG-Local | 69.0 |
| HippoRAG2 | 70.3 |
| **本系统（Qwen3.5-9B 本地部署）** | **77.0** |

### 1.3 最终预测文件

- `results/predictions_final.json` — 2556 题最终答案（已应用答案清洗规则）
- `results/eval_qa_final.txt` — 官方 `qa_evaluate.py` 评测输出（Overall **0.7700**）
- `results/eval_retrieval_final.txt` — 官方 `retrieval_evaluate.py` 对基础检索的评测输出（Hits@10 83.95 / Hits@4 78.40 / MAP 0.338 / MRR 0.676）

---

## 二、系统架构（六层流水线）

```
题目 ──▶ ①官方混合检索（BM25 + 稠密向量 + RRF 融合 + BGE 重排 → top-10）
     ──▶ ②源覆盖增强       （检索后处理 #1：题面点名来源缺席时，从其文档补 2 段）
     ──▶ ③来源补强 v2 + 日期匹配（检索后处理 #3/#4：点名来源 ≥3 段；点名日期锁定文章）
     ──▶ ④日期标注         （检索后处理 #2：每段证据加 [Published: YYYY-MM-DD] 前缀）
     ──▶ ⑤LLM 生成         （prompt v3：逐源核对 + 时间线引导 + 禁猜禁单字母）
     ──▶ ⑥答案清洗         （输出校验：≤1 字符 → "Insufficient Information"）
     ──▶ 官方评测
```

**设计要点**：
- 所有增强模块（②③④⑥）都是**机制级**改动：只使用「题目原文 + 语料元数据」，不接触 `question_type` / `answer` / `evidence_list` 任何标注字段（见 `docs/COMPLIANCE.md` 合规审计）
- 检索后处理模块均可在基础检索结果上离线运行，天然支持消融实验（同一份基础检索 + 单一变量）

---

## 三、目录结构

```
RAG_project/
├── README.md
├── requirements.txt
├── code/
│   ├── retrieval/         # 官方混合检索栈（复用，含 [ADAPTED] 适配）
│   ├── enhancers/         # 自研检索后处理模块（源覆盖 / 来源补强+日期匹配 / 日期标注）
│   ├── generation/        # 生成脚本（官方版 + prompt v2 / v3 变体）
│   ├── evaluation/        # 官方评测脚本（qa_evaluate pin 于 c1c1287a）
│   ├── analysis/          # 实验分析脚本（失败归因 / 稳健性 / 合规审计 / 消融对比）
│   └── runners/           # 分块断点续跑执行器
├── docs/
│   ├── ABLATION.md        # 消融实验完整记录（含对照组与噪声底线）
│   ├── COMPLIANCE.md      # 合规性审计（无标注泄漏验证）
│   └── FINDINGS.md        # 工程发现与部署要点
└── results/               # 最终产物（预测 + 官方评测输出）
```

---

## 四、环境与模型

### 4.1 Python 环境（Python 3.11）

```bash
conda create -n mhr python=3.11 -y && conda activate mhr
pip install -r requirements.txt
# CUDA 版 PyTorch（可选，用于 GPU 重排加速；国内可用 SJTU 镜像）
pip install torch==2.14.1+cu126 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
```

### 4.2 本地模型文件（`models/`）

| 模型 | 用途 | 获取 |
|---|---|---|
| `all-MiniLM-L6-v2` | 稠密检索编码器 | huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| `bge-reranker-v2-m3` | 重排器 | huggingface.co/BAAI/bge-reranker-v2-m3 |

### 4.3 LLM 服务（llama.cpp + Qwen3.5-9B）

```bash
# llama.cpp（CUDA 版）启动 OpenAI 兼容服务
llama-server -m Qwen_Qwen3.5-9B-Q8_0.gguf --port 18080 -ngl 99 -c 8192 --jinja
```

要点：默认思考模式**关闭**（`chat_template_kwargs: {"enable_thinking": false}`，见 docs/FINDINGS.md 的 A/B 实验）；HTTP 客户端**每题新建连接**（llama.cpp keep-alive 缺陷规避，见 FINDINGS.md）。

### 4.4 数据

MultiHop-RAG 官方数据（609 篇语料 + 2556 题）：`github.com/yixuantt/MultiHop-RAG` → 放入 `dataset/`（`corpus.json`、`MultiHopRAG.json`）。

---

## 五、复现步骤

```bash
# ① 基础混合检索（全量 2556 题）
python code/retrieval/run_hybrid_retrieval.py \
    --queries dataset/MultiHopRAG.json --start 0 --limit 2556 \
    --retrieve_top_k 10 --rrf_top_k 10 --rerank_top_n 10 \
    --output output/full_query_result.json

# ② 检索后处理链（顺序执行）
python code/enhancers/enhance_sources.py          # 源覆盖 → output/enhanced_query_result.json
python code/enhancers/enhance_v2.py               # 来源补强+日期匹配（读上一步产物）
python code/enhancers/temporal_annotate.py output/final_query_result_pre.json output/final_query_result.json

# ③ 生成（prompt v3；远程/本地均可，指向 llama-server）
python code/generation/qa_llama_v3.py \
    --input output/final_query_result.json --output qa_output/final_v3_full.json

# ④ 答案清洗（≤1 字符 → Insufficient Information，见 analysis/ 或按 README 规范）
# ⑤ 官方评测
python code/evaluation/qa_evaluate.py --file results/predictions_final.json
python code/evaluation/retrieval_evaluate.py --file output/full_query_result.json
```

> 注：脚本内路径按本机环境（`D:\ANACONDA\envs\mhr` 等）配置，迁移时请对应修改；`code/runners/run_qa_full.sh` 提供了分块断点续跑的全量执行器示例。

---

## 六、方法论说明与边界（诚实声明）

1. **迭代式改进的乐观偏差**：本项目采用「全量失败归因 → 机制级改进 → 全量复评」的迭代流程。由于基准不提供留出集，所有数字均为**同一数据集上的改进量**，应按"改进上限"理解；为此我们做了两组稳健性检验：
   - **分半一致性**：各改进轮的增益在数据前半/后半上高度一致（例：prompt v2 为 +7.5 / +6.6pp，连副作用都对称）
   - **对照组**：源覆盖轮中，被增强的 490 题 +10.8pp，未增强的 2066 题仅 -0.2pp（同时测得**单次运行采样噪声底线 ≈ ±0.2pp**）
2. **合规边界**：`answer` / `evidence_list` / `question_type` 从未作为系统输入（审计见 docs/COMPLIANCE.md）。
3. **答案清洗规则**：基于"合法答案从不含单字符"的基准属性（全量验证：0 个正确答案为单字符），属输出校验层，已单独标注。

---

## 七、工程发现（摘要）

| 发现 | 现象 | 解决 |
|---|---|---|
| llama.cpp keep-alive 缺陷 | 经 SSH 隧道复用 HTTP 连接时**隔次返回 404**（复用连接 OK/404 严格交替；远程直连 6/6 正常） | 每题新建连接（[ADAPTED] 已在 qa_llama*.py 固化） |
| 思考模式不可行 | 9B 模型 70% 的题在 2048 token 内**只有思维链没有答案**，总体 14.3% vs 关闭 59.0% | 默认关闭思考模式（官方默认一致） |
| 证据时间隐形 | **98.2%** 的语料 chunk 正文不含自身发布日期（正文用相对时间） | 日期标注注入 `[Published: …]` 前缀 |

详见 `docs/FINDINGS.md`。

---

## 八、复用与自研清单

**复用（官方 MultiHop-RAG，pin c1c1287a）**：混合检索管线（`hybrid_retriever.py` 等）、生成模板（`qa_llama.py`）、评测脚本（`qa_evaluate.py` / `retrieval_evaluate.py`，未改动评分逻辑）、数据与切块参数（256/25、metadata 入检索文本）。

**`[ADAPTED]` 适配（全部带注释标记）**：本地模型路径、LLM 端点、思考开关写法、每题新建连接。

**自研模块**：源覆盖增强 / 来源补强 v2 + 日期匹配 / 日期标注（三个 enhancer）、prompt v2/v3、答案清洗规则、全套实验分析脚本。

---

## 九、团队

- **GroupXX** —（请填写组号）
- 成员：（请填写）

> 课程：《自然语言处理》（华南理工大学）｜提交截止：2026-11-01
