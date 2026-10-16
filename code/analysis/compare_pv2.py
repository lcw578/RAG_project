# -*- coding: utf-8 -*-
"""prompt v2 vs v1 全量对比：分题型准确率 + 规则叠加效果"""
import sys, io, json
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

def apply_rule(ans):
    a = ans.strip()
    if len(a) <= 1:
        return "Insufficient Information"
    return ans

v1 = json.load(open(r'qa_output/full_qwen.json', encoding='utf-8'))
v2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
print(f"v1 records: {len(v1)}, v2 records: {len(v2)}")

agg = defaultdict(lambda: {'v1': 0, 'v1r': 0, 'v2': 0, 'v2r': 0, 'n': 0})
flips_12 = 0
ii_v1 = defaultdict(int)   # comparison 中答 Insufficient Information 的数量
ii_v2 = defaultdict(int)
for a, b in zip(v1, v2):
    t = a['question_type']
    agg[t]['n'] += 1
    a1 = strict(a['model_answer'], a['gold_answer'])
    a1r = strict(apply_rule(a['model_answer']), a['gold_answer'])
    a2 = strict(b['model_answer'], b['gold_answer'])
    a2r = strict(apply_rule(b['model_answer']), b['gold_answer'])
    agg[t]['v1'] += a1; agg[t]['v1r'] += a1r
    agg[t]['v2'] += a2; agg[t]['v2r'] += a2r
    if a1 != a2:
        flips_12 += 1
    for typ in (t,):
        if 'insufficient' in a['model_answer'].lower():
            ii_v1[typ] += 1
        if 'insufficient' in b['model_answer'].lower():
            ii_v2[typ] += 1

print(f"\n{'题型':<20}{'v1':>8}{'v1+规则':>10}{'v2':>8}{'v2+规则':>10}{'n':>6}")
T = {'v1': 0, 'v1r': 0, 'v2': 0, 'v2r': 0, 'n': 0}
for t, v in sorted(agg.items()):
    print(f"{t:<20}{v['v1']/v['n']*100:>7.1f}%{v['v1r']/v['n']*100:>9.1f}%{v['v2']/v['n']*100:>7.1f}%{v['v2r']/v['n']*100:>9.1f}%{v['n']:>6}")
    for k in T: T[k] += v[k]
print(f"{'总体':<20}{T['v1']/T['n']*100:>7.1f}%{T['v1r']/T['n']*100:>9.1f}%{T['v2']/T['n']*100:>7.1f}%{T['v2r']/T['n']*100:>9.1f}%{T['n']:>6}")
print(f"\nv1→v2 翻转题数: {flips_12}")
print(f"答 II 数量对比（v1 → v2）:")
for t in sorted(set(list(ii_v1.keys()) + list(ii_v2.keys()))):
    print(f"  {t}: {ii_v1[t]} -> {ii_v2[t]}")
# v2 单字符垃圾
garbage_v2 = sum(1 for b in v2 if len(b['model_answer'].strip()) <= 1)
print(f"\nv2 单字符答案: {garbage_v2}（v1: 85）")
