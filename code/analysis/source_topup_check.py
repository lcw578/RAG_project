# -*- coding: utf-8 -*-
"""量化：comparison/temporal 题里"点名来源在 context 中的占比"分布
   —— 来源补强（把每来源的 chunk 数从 ≥2 提到 ≥3）能覆盖的池子"""
import sys, io, json, re
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

def rule(ans):
    return "Insufficient Information" if len(ans.strip()) <= 1 else ans

ret = json.load(open(r'output/enhanced_temporal_query_result.json', encoding='utf-8'))
v3 = json.load(open(r'qa_output/v3_full.json', encoding='utf-8'))

# 重建来源匹配（与 enhance_sources.py 同逻辑）
import json as _j
raw = _j.load(open(r'storage/all-MiniLM-L6-v2_chunk256/docstore.json', encoding='utf-8'))
data = raw.get('docstore/data', raw)
sources = set()
for _, w in data.items():
    d = w.get('__data__', w)
    s = str((d.get('metadata') or {}).get('source', '')).strip()
    if s:
        sources.add(s)

def keys_of(s):
    ks = set()
    if len(s) >= 4: ks.add(s)
    first = s.split('|')[0].strip()
    if len(first) >= 4: ks.add(first)
    return ks

key_to_source = {}
for s in sources:
    for k in keys_of(s):
        key_to_source.setdefault(k, s)
key_pat = [(k, re.compile(r'(?<![A-Za-z0-9])' + re.escape(k) + r'(?![A-Za-z0-9])')) for k in sorted(key_to_source, key=len, reverse=True)]

dist = defaultdict(Counter)   # 题型 -> 每来源chunk数分布
fail_dist = defaultdict(Counter)  # 只统计失败的题
for a, r in zip(v3, ret):
    t = a['question_type']
    if t not in ('comparison_query', 'temporal_query'):
        continue
    wanted = []
    for k, pat in key_pat:
        if pat.search(r['query']):
            s = key_to_source[k]
            if s not in wanted:
                wanted.append(s)
    if len(wanted) < 2:
        continue
    cnt = Counter(m.get('source') for m in r['retrieval_list'])
    per_source = [cnt.get(s, 0) for s in wanted]
    ok = strict(rule(a['model_answer']), a['gold_answer'])
    # 特征：最弱来源的 chunk 数
    weakest = min(per_source)
    dist[t][weakest] += 1
    if not ok:
        fail_dist[t][weakest] += 1

for t in ['comparison_query', 'temporal_query']:
    print(f"=== {t}（点名 ≥2 来源的题）===")
    tot = sum(dist[t].values())
    print(f"  总数 {tot}")
    for w in sorted(dist[t]):
        n = dist[t][w]; f = fail_dist[t][w]
        print(f"  最弱来源只有 {w} 段: {n} 题（其中失败 {f}，失败率 {f/n*100:.1f}%）")
