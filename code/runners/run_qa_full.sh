#!/bin/bash
# 全量 QA 分块流水线（稳妥模式 v2）：
#   分块 -> 逐块生成（断点续跑 + 每块前等隧道恢复 + 最多 3 次尝试）
#   -> 按序合并 -> 官方评测（答案准确率 + 检索指标）
# 用法：bash run_qa_full.sh
# 依赖：output/full_query_result.json；远程 llama-server 在跑；本机隧道守护任务在跑（自动重连）。
cd "$(dirname "$0")"
PY="D:/ANACONDA/envs/mhr/python.exe"
IN="output/full_query_result.json"
CHUNK_DIR="output/qa_chunks"
OUT_DIR="qa_output/chunks"
MERGED="qa_output/full_qwen.json"
CHUNK=300
mkdir -p "$CHUNK_DIR" "$OUT_DIR" logs

wait_tunnel() {
  local k ok=0
  for k in $(seq 1 30); do
    if curl -s -m 4 http://127.0.0.1:18080/health 2>/dev/null | grep -q '"ok"'; then
      ok=1
      return 0
    fi
    sleep 5
  done
  return 1
}

if [ ! -s "$IN" ]; then
  echo "### $IN 不存在或为空，先跑完检索再执行本脚本 ###"
  exit 1
fi

echo "########## [1/4] 切分输入 ##########"
"$PY" - "$IN" "$CHUNK_DIR" "$CHUNK" <<'EOF'
import json, sys, math
src, outdir, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(src, encoding='utf-8'))
n = math.ceil(len(data) / chunk)
for i in range(n):
    part = data[i * chunk:(i + 1) * chunk]
    json.dump(part, open(f"{outdir}/in_{i:02d}.json", 'w', encoding='utf-8'), ensure_ascii=False)
print(f"split {len(data)} records into {n} chunks of <= {chunk}")
EOF

N=$(ls "$CHUNK_DIR"/in_*.json | wc -l)
echo "chunks: $N"

echo "########## [2/4] 逐块生成（断点续跑 + 隧道等待）##########"
for i in $(seq 0 $((N - 1))); do
  ii=$(printf "%02d" "$i")
  in="$CHUNK_DIR/in_$ii.json"
  out="$OUT_DIR/out_$ii.json"
  if [ -s "$out" ]; then
    echo "[skip] chunk $ii 已完成"
    continue
  fi

  echo "[wait] chunk $ii：确认隧道/服务 ... $(date '+%H:%M:%S')"
  if ! wait_tunnel; then
    echo "### [FAIL] chunk $ii：隧道 150 秒内未恢复，中止。重跑本脚本可断点续跑 ###"
    exit 1
  fi

  rc=1
  for attempt in 1 2 3; do
    if [ "$attempt" -gt 1 ]; then
      echo "[retry] chunk $ii 第 $attempt 次尝试（20 秒后）... $(date '+%H:%M:%S')"
      sleep 20
      wait_tunnel || { echo "### [FAIL] 隧道仍不可用，中止 ###"; exit 1; }
    fi
    echo "[run ] chunk $ii attempt $attempt $(date '+%H:%M:%S') ..."
    "$PY" qa_llama.py --input "$in" --output "$out" >> "logs/qa_chunk_$ii.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$out" ]; then
      break
    fi
  done

  if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
    echo "### [FAIL] chunk $ii 三次尝试均失败（exit=$rc），中止。日志：logs/qa_chunk_$ii.log ###"
    exit 1
  fi
  echo "[done] chunk $ii OK $(date '+%H:%M:%S')"
done

echo "########## [3/4] 合并 ##########"
"$PY" - "$OUT_DIR" "$MERGED" "$N" <<'EOF'
import json, sys
outdir, merged, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
all_items = []
for i in range(n):
    part = json.load(open(f"{outdir}/out_{i:02d}.json", encoding='utf-8'))
    all_items.extend(part)
json.dump(all_items, open(merged, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"merged {len(all_items)} answers -> {merged}")
EOF

echo "########## [4/4] 官方评测 ##########"
echo "----- 答案准确率（全量）-----"
"$PY" qa_evaluate.py --file "$MERGED"
echo "----- 检索指标（全量）-----"
"$PY" retrieval_evaluate.py --file "output/full_query_result.json"
echo "### QA FULL PIPELINE DONE ###"
