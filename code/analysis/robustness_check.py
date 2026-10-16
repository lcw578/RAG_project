# -*- coding: utf-8 -*-
"""v2 稳健性检验：分半一致性 + 翻转的分布均匀性"""
import sys, io, json
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

v1 = json.load(open(r'qa_output/full_qwen.json', encoding='utf-8'))
v2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
N = len(v1)
print(f"总数: {N}")

# ===== 1) 分半一致性 =====
print("\n===== 分半对比（前半 / 后半）=====")
for half_name, lo, hi in [("前半", 0, N // 2), ("后半", N // 2, N)]:
    agg = defaultdict(lambda: {'v1': 0, 'v2': 0, 'n': 0})
    for a, b in zip(v1[lo:hi], v2[lo:hi]):
        t = a['question_type']
        agg[t]['n'] += 1
        agg[t]['v1'] += strict(a['model_answer'], a['gold_answer'])
        agg[t]['v2'] += strict(b['model_answer'], b['gold_answer'])
    print(f"[{half_name}]")
    tv1 = tv2 = tn = 0
    for t, v in sorted(agg.items()):
        print(f"  {t:<20} v1={v['v1']/v['n']*100:>5.1f}%  v2={v['v2']/v['n']*100:>5.1f}%  Δ={((v['v2']-v['v1'])/v['n'])*100:>+5.1f}  n={v['n']}")
        tv1 += v['v1']; tv2 += v['v2']; tn += v['n']
    print(f"  {'总体':<20} v1={tv1/tn*100:>5.1f}%  v2={tv2/tn*100:>5.1f}%  Δ={((tv2-tv1)/tn)*100:>+5.1f}")

# ===== 2) 翻转按数据段的分布 =====
print("\n===== 每题净变化(答对=1/答错=0) 按 10 段分布的均值 =====")
seg = 10
step = N // seg
net_by_seg = []
for s in range(seg):
    lo, hi = s * step, (s + 1) * step if s < seg - 1 else N
    net = 0
    for a, b in zip(v1[lo:hi], v2[lo:hi]):
        net += strict(b['model_answer'], b['gold_answer']) - strict(a['model_answer'], a['gold_answer'])
    net_by_seg.append(net)
    print(f"  段{s+1:>2} [{lo:>4}-{hi:>4}]: 净变化 {net:+d}  ({net/(hi-lo)*100:+.1f}pp)")

# ===== 3) 增益的题型构成 =====
print("\n===== 总净增益的构成 =====")
by_type = defaultdict(int)
for a, b in zip(v1, v2):
    by_type[a['question_type']] += strict(b['model_answer'], b['gold_answer']) - strict(a['model_answer'], a['gold_answer'])
total = sum(by_type.values())
for t, v in sorted(by_type.items()):
    print(f"  {t:<20} {v:+d} 题  ({v/total*100:>5.1f}% of 净增)")
print(f"  合计净增: {total:+d} 题")
