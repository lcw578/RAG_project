# -*- coding: utf-8 -*-
"""官方文件无 question_id，按数组顺序生成 q0000..q2555，再按 query 映射到预测。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "dataset" / "MultiHopRAG.json"
PRED = ROOT / "results" / "predictions_final.json"
OUT = ROOT / "results" / "predictions.json"

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

with OFFICIAL.open("r", encoding="utf-8") as f:
    official = json.load(f)

if not isinstance(official, list):
    raise SystemExit("官方文件不是数组")

query2id = {}
for idx, item in enumerate(official):
    q = norm(item["query"])
    if q in query2id:
        print("警告：官方重复 query，index", idx, "已有", query2id[q])
    else:
        query2id[q] = f"q{idx:04d}"

with PRED.open("r", encoding="utf-8") as f:
    preds = json.load(f)

out, missing = [], []
for p in preds:
    qid = query2id.get(norm(p["query"]))
    if qid is None:
        missing.append(p["query"][:80])
        continue
    out.append({"question_id": qid, "model_answer": p["model_answer"]})

out.sort(key=lambda x: x["question_id"])
expected = [f"q{i:04d}" for i in range(len(official))]
got = [x["question_id"] for x in out]

print("官方条数:", len(official))
print("预测条数:", len(preds))
print("输出条数:", len(out))
print("缺失条数:", len(missing))
print("ID 完全匹配:", got == expected)
print("重复 ID:", len(got) - len(set(got)))

if missing:
    print("\n匹配不上的前 5 条 query：")
    for m in missing[:5]:
        print(" -", m)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(f"\n已写入: {OUT}")