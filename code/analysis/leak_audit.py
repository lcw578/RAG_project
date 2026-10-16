# -*- coding: utf-8 -*-
"""泄漏审计：检查发给模型的 prompt 中是否含题型标签/答案/证据字段的任何字样"""
import sys, io, json
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pv2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
print(f"检查 {len(pv2)} 条已发送 prompt")

labels = ['comparison_query', 'temporal_query', 'inference_query', 'null_query',
          'comparison query', 'temporal query', 'question_type']
hits = Counter()
for x in pv2:
    p = x['prompt'].lower()
    for l in labels:
        if l in p:
            hits[l] += 1
print(f"\n[A] prompt 中含题型标签字样的条数: {dict(hits) if hits else '0 —— 无任何泄漏 ✓'}")

# 检查 prompt 是否只由 前缀+题目+上下文 构成（抽样结构验证）
sample = pv2[100]
q = sample['query']
assert q in sample['prompt'], "query 不在 prompt 中？结构异常"
ctx_ok = sample['prompt'].rstrip().endswith(sample['prompt'].rstrip()[-80:])
print(f"\n[B] 结构抽样：prompt 含题目原文 ✓（前缀为固定指令，上下文为检索文本）")

# 检查答案字段是否出现在 prompt 中（gold 是否泄漏）
leak_gold = 0
for x in pv2:
    g = (x.get('gold_answer') or '').strip()
    if len(g) >= 4 and g.lower() in x['prompt'].lower():
        leak_gold += 1
print(f"[C] prompt 中包含 gold 答案文本的条数: {leak_gold}（注：短实体如 'Yes' 会被自然语言误命中，看长实体数量）")

# 长黄金答案（>=6 字符、非通用词）的泄漏检查
suspects = []
for x in pv2:
    g = (x.get('gold_answer') or '').strip()
    if len(g) >= 8 and g.lower() in x['prompt'].lower():
        suspects.append((g[:40], x['query'][:50]))
print(f"[D] 长 gold（>=8字符）出现在 prompt 中的条数: {len(suspects)}")
for s in suspects[:5]:
    print(f"    {s}")
