# -*- coding: utf-8 -*-
"""最终版对比：v1 → v2 → 增强 → v3 → 最终(补强v2+日期匹配)"""
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
fin = json.load(open(r'qa_output/final_v3_full.json', encoding='utf-8'))
assert len(v1) == len(v2) == len(enh) == len(v3) == len(fin)

agg = defaultdict(lambda: {'v1': 0, 'v2': 0, 'enh': 0, 'v3': 0, 'fin': 0, 'finr': 0, 'n': 0})
for a, b, c, d, e in zip(v1, v2, enh, v3, fin):
    t = a['question_type']
    agg[t]['n'] += 1
    agg[t]['v1'] += strict(a['model_answer'], a['gold_answer'])
    agg[t]['v2'] += strict(b['model_answer'], b['gold_answer'])
    agg[t]['enh'] += strict(c['model_answer'], c['gold_answer'])
    agg[t]['v3'] += strict(d['model_answer'], d['gold_answer'])
    agg[t]['fin'] += strict(e['model_answer'], e['gold_answer'])
    agg[t]['finr'] += strict(rule(e['model_answer']), e['gold_answer'])

print(f"{'题型':<20}{'v1':>7}{'v2':>7}{'增强':>7}{'v3':>7}{'最终':>7}{'最终+规则':>10}{'n':>6}")
T = {'v1': 0, 'v2': 0, 'enh': 0, 'v3': 0, 'fin': 0, 'finr': 0, 'n': 0}
for t, v in sorted(agg.items()):
    print(f"{t:<20}{v['v1']/v['n']*100:>6.1f}%{v['v2']/v['n']*100:>6.1f}%{v['enh']/v['n']*100:>6.1f}%{v['v3']/v['n']*100:>6.1f}%{v['fin']/v['n']*100:>6.1f}%{v['finr']/v['n']*100:>9.1f}%{v['n']:>6}")
    for k in T: T[k] += v[k]
print(f"{'总体':<20}{T['v1']/T['n']*100:>6.1f}%{T['v2']/T['n']*100:>6.1f}%{T['enh']/T['n']*100:>6.1f}%{T['v3']/T['n']*100:>6.1f}%{T['fin']/T['n']*100:>6.1f}%{T['finr']/T['n']*100:>9.1f}%{T['n']:>6}")

# v3 → 最终：翻转分析
print("\n=== v3 → 最终 翻转分析 ===")
for t in ['comparison_query', 'temporal_query', 'inference_query', 'null_query']:
    up = down = 0
    for d, e in zip(v3, fin):
        if d['question_type'] != t:
            continue
        a = strict(d['model_answer'], d['gold_answer'])
        b = strict(e['model_answer'], e['gold_answer'])
        if b and not a: up += 1
        if a and not b: down += 1
    print(f"  {t:<20} 翻对 +{up}, 翻错 -{down}, 净 {up-down:+d}")
