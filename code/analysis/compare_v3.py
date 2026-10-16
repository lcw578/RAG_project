# -*- coding: utf-8 -*-
"""v3（日期标注+时间引导）最终对比：v1 → v2 → 增强 → v3"""
import sys, io, json
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

def rule(ans):
    return "Insufficient Information" if len(ans.strip()) <= 1 else ans

v1 = json.load(open(r'qa_output/full_qwen.json', encoding='utf-8'))
v2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
enh = json.load(open(r'qa_output/enh_v2_full.json', encoding='utf-8'))
v3 = json.load(open(r'qa_output/v3_full.json', encoding='utf-8'))
assert len(v1) == len(v2) == len(enh) == len(v3)

agg = defaultdict(lambda: {'v1': 0, 'v2': 0, 'enh': 0, 'v3': 0, 'v3r': 0, 'n': 0})
for a, b, c, d in zip(v1, v2, enh, v3):
    t = a['question_type']
    agg[t]['n'] += 1
    agg[t]['v1'] += strict(a['model_answer'], a['gold_answer'])
    agg[t]['v2'] += strict(b['model_answer'], b['gold_answer'])
    agg[t]['enh'] += strict(c['model_answer'], c['gold_answer'])
    agg[t]['v3'] += strict(d['model_answer'], d['gold_answer'])
    agg[t]['v3r'] += strict(rule(d['model_answer']), d['gold_answer'])

print(f"{'题型':<20}{'v1':>7}{'v2':>7}{'增强':>7}{'v3':>7}{'v3+规则':>9}{'n':>6}")
T = {'v1': 0, 'v2': 0, 'enh': 0, 'v3': 0, 'v3r': 0, 'n': 0}
for t, v in sorted(agg.items()):
    print(f"{t:<20}{v['v1']/v['n']*100:>6.1f}%{v['v2']/v['n']*100:>6.1f}%{v['enh']/v['n']*100:>6.1f}%{v['v3']/v['n']*100:>6.1f}%{v['v3r']/v['n']*100:>8.1f}%{v['n']:>6}")
    for k in T: T[k] += v[k]
print(f"{'总体':<20}{T['v1']/T['n']*100:>6.1f}%{T['v2']/T['n']*100:>6.1f}%{T['enh']/T['n']*100:>6.1f}%{T['v3']/T['n']*100:>6.1f}%{T['v3r']/T['n']*100:>8.1f}%{T['n']:>6}")

# temporal 单独深挖：v3 vs 增强
print("\n=== temporal（583 题）v3 vs 增强 翻转分析 ===")
tc = te = 0
up = down = 0
for e, d in zip(enh, v3):
    if e['question_type'] != 'temporal_query':
        continue
    a = strict(e['model_answer'], e['gold_answer'])
    b = strict(d['model_answer'], d['gold_answer'])
    tc += a; te += b
    if b and not a: up += 1
    if a and not b: down += 1
print(f"  增强: {tc} → v3: {te}  (Δ={te-tc:+d} 题)")
print(f"  翻对: +{up}  翻错: -{down}")
# 非 temporal 组对照（v3 的副作用检查）
nochange = 0; total_other = 0
for e, d in zip(enh, v3):
    if e['question_type'] == 'temporal_query':
        continue
    total_other += 1
    nochange += (strict(e['model_answer'], e['gold_answer']) != strict(d['model_answer'], d['gold_answer']))
print(f"\n非 temporal 组（{total_other} 题）结果变化数: {nochange}（含采样噪声）")
