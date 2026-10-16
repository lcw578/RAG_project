# 坑位手册（症状 → 原因 → 解决）

> 全部为本项目实际踩过并验证修复的坑；每条附验证方法。

## 1. `No module named 'pkg_resources'`
- **症状**：`llama_index` 导入失败，提示 pkg_resources 缺失
- **原因**：llama-index 0.9.x 依赖 `pkg_resources`，而 setuptools ≥ 81 已将其移除
- **解决**：`pip install "setuptools<81"`（requirements.txt 已含）
- **验证**：`python -c "import llama_index; print('ok')"`

## 2. hf download 报 401（hf-mirror）
- **症状**：`hf download` 走 hf-mirror 时返回 401 Unauthorized
- **原因**：新版 huggingface_hub 默认启用 Xet 传输协议，hf-mirror 不支持
- **解决**：`export HF_HUB_DISABLE_XET=1`（或改用 curl 直连下载）
- **验证**：重新执行下载，观察进度条正常推进

## 3. torch 安装失败 / 清华 pytorch-wheels 404
- **症状**：从清华镜像装 CUDA 版 torch 报 404
- **原因**：清华 pytorch-wheels 镜像已下线（实测 2026-10）
- **解决**：改用 SJTU 镜像：
  `pip install torch==2.14.1+cu126 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126`
- **验证**：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

## 4. llama-server 端口冲突
- **症状**：llama-server 启动失败或连不上
- **原因**：8080 常被代理类软件（clash 等）占用
- **解决**：统一用 **18080**：`llama-server ... --port 18080`
- **验证**：`curl http://127.0.0.1:18080/health` 返回 `{"status":"ok"}`

## 5. 模型只输出思维链、没有答案
- **症状**：响应含 `reasoning_content`，`content` 为空；或 max_tokens 被思考耗尽（本项目实测：2048 token 下 70% 的题只有思考没有答案）
- **原因**：Qwen3.5 系列默认开启思考模式
- **解决**：客户端请求传 `chat_template_kwargs: {"enable_thinking": false}`（本项目 `qa_llama*.py` 已内置）
- **验证**：同一 prompt 对比开启/关闭时 `usage.completion_tokens` 与 content

## 6. 复用连接调用「隔次 404」
- **症状**：复用同一 HTTP 连接时，请求 OK/404/OK/404 严格交替（100% 复现）；每题新建连接则完全正常
- **原因**：llama.cpp（b11429）+ keep-alive 的兼容性缺陷（对照实验见项目 docs/FINDINGS.md）
- **解决**：**每请求新建连接**（`qa_llama*.py` 的 `[ADAPTED]` 已固化；本机建连毫秒级，无性能损失）
- **验证**：同一脚本对连续 10 个请求统计 HTTP 状态码

## 7. 机器卡死（内存）
- **症状**：加载模型相关脚本运行时整机无响应
- **原因**：16GB 内存机器上同时加载两份 2.3GB 重排模型（+系统占用）→ 内存打爆触发换页风暴
- **解决**：同一脚本绝不重复加载大模型；加载前先确认可用内存（>4GB）
- **验证**：`tasklist` 观察 python 进程内存占用峰值

## 8. GPU 重排是否生效
- **现象**：FlagEmbedding 构造时 `model.device` 显示 cpu，易误判
- **事实**：新版 FlagEmbedding 在**首次推理**时自动迁移到 CUDA——用显存占用判断更可靠（约 +2.3GB）
- **验证**：`torch.cuda.memory_allocated()` 在首次 `compute_score` 前后对比

## 9. 中文路径 / 编码相关
- **症状**：个别工具链在中文路径下行为异常；Windows 控制台输出中文乱码
- **解决**：项目/模型路径尽量 ASCII；Python 脚本头加 `# -*- coding: utf-8 -*-` 并在输出时包 `io.TextIOWrapper(...encoding='utf-8')`
- **验证**：脚本重定向输出到文件后检查无乱码
