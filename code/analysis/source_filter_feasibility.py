# -*- coding: utf-8 -*-
"""源过滤可行性验证：语料来源名单 × 题面点名率"""
import sys, io, json, re
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

corpus = json.load(open(r'dataset/corpus.json', encoding='utf-8'))
queries = json.load(open(r'dataset/MultiHopRAG.json', encoding='utf-8'))

# 语料来源名单（去重、去空）
sources = sorted(set(str(d.get('source', '')).strip() for d in corpus if d.get('source')))
print(f"语料来源数: {len(sources)}")
print("样例:", sources[:12])

# 为每个来源建纯小写匹配键（保留可辨识的全名）
def mentioned_sources(q, sources):
    ql = q.lower()
    hits = []
    for s in sources:
        sl = s.lower()
        if len(sl) >= 4 and sl in ql:
            hits.append(s)
    return hits

mention_dist = Counter()
by_type = {}
examples = []
for item in queries:
    hits = mentioned_sources(item['query'], sources)
    k = min(len(hits), 3)
    mention_dist[k] += 1
    t = item['question_type']
    by_type.setdefault(t, Counter())[k] += 1
    if t == 'comparison_query' and len(hits) >= 2 and len(examples) < 5:
        examples.append((item['query'][:130], hits))

print("\n=== 全局：题面提到的语料来源数量分布 ===")
for k in sorted(mention_dist):
    print(f"  提到 {k} 个: {mention_dist[k]} 题")

print("\n=== 按题型 ===")
for t in ['comparison_query', 'temporal_query', 'inference_query', 'null_query']:
    c = by_type.get(t, Counter())
    n = sum(c.values())
    ge2 = c.get(2, 0) + c.get(3, 0)
    print(f"  {t:<18} n={n:<4}  提到≥2个来源: {ge2} ({ge2/n*100:.0f}%)  提到≥1个: {n - c.get(0, 0)} ({(n-c.get(0,0))/n*100:.0f}%)")

print("\n=== comparison 样例（题面 → 匹配到的来源）===")
for q, hits in examples:
    print(f"  Q: {q}...")
    print(f"     → {hits}")
