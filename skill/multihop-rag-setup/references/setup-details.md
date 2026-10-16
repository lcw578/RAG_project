# 环境配置详细参考

## 一、组件清单与体积

| 组件 | 体积 | 说明 |
|---|---|---|
| conda 环境 mhr（py3.11 + 依赖 + CUDA torch） | ~6 GB | 见 requirements.txt |
| all-MiniLM-L6-v2 | ~90 MB | 稠密检索编码器 |
| bge-reranker-v2-m3 | ~2.3 GB | 重排器（fp32 显存占用同量级） |
| Qwen3.5-9B Q8_0 GGUF | ~9.8 GB | 生成模型（llama.cpp 服务） |
| MultiHop-RAG 数据 | ~80 MB | corpus.json + MultiHopRAG.json |
| 运行产物（索引 + 结果） | ~1 GB | storage/ output/ qa_output/ |

## 二、模型下载渠道

| 模型 | 渠道 | 命令要点 |
|---|---|---|
| all-MiniLM-L6-v2 | hf-mirror | `HF_HUB_DISABLE_XET=1 hf download sentence-transformers/all-MiniLM-L6-v2 ...` |
| bge-reranker-v2-m3 | hf-mirror | 同上，加 `--exclude "onnx/*"` 省流量 |
| Qwen3.5-9B Q8_0 | ModelScope 或 hf-mirror（bartowski 量化） | 官方 GGUF 仓库 ModelScope 不存在，用 bartowski 版 |

GitHub 大文件下载不稳时：走本机代理 + `scp -O` 中继到目标机器（本项目实践）；curl 需 `--ssl-no-revoke`（schannel 吊销检查问题）。

## 三、llama.cpp 服务部署（两种形态）

### A. 本机运行（显存 ≥ 11GB）

```bash
llama-server -m Qwen_Qwen3.5-9B-Q8_0.gguf --port 18080 -ngl 99 -c 8192 --jinja
```

### B. 局域网另一台机器运行（本项目实际采用：笔记本无大显存）

1. 在 GPU 机器上以同样命令启动服务
2. 客户端侧通过 SSH 隧道映射端口：`ssh -N -L 18080:127.0.0.1:18080 <user>@<gpu-host>`
3. 客户端代码只需 `LOCAL_LLM_BASE_URL=http://127.0.0.1:18080/v1`（默认值）
4. 注意：隧道场景必须规避 keep-alive 缺陷（见 pitfalls.md #6）

### 服务端关键参数

| 参数 | 值 | 原因 |
|---|---|---|
| `-ngl` | 99 | 全部层上 GPU |
| `-c` | 8192 | 上下文长度（本项目单题 prompt ≈ 5-6k token） |
| `--jinja` | 开 | 启用聊天模板（否则 chat_template_kwargs 不生效） |
| 端口 | 18080 | 8080 常被占用 |

## 四、索引与产物说明（避免重复劳动）

- 首次检索运行时自动构建并持久化索引到 `storage/`（约 90MB）；后续运行直接复用（冷启动 ~40s → 热启动 ~10s）
- 检索结果、生成答案、评测输出分别落在 `output/`、`qa_output/`
- 全量 2556 题：检索约 40-55 分钟（GPU 重排）、生成约 22 分钟（关思考、远程服务）、评测 < 1 分钟

## 五、可选优化

- GPU 重排：装 CUDA 版 torch 后自动生效（无需改代码）
- 批量运行：`code/runners/run_qa_full.sh`（分块 + 断点续跑 + 失败重试）
- 远程隔离执行：将生成脚本部署到 LLM 所在机器，以「计划任务/服务」方式脱离会话运行（本项目实践：SYSTEM 账户计划任务 + 日志重定向）
