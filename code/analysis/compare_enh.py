# -*- coding: utf-8 -*-
"""最终对比：v1基线 → v2(prompt) → v2+源增强（含 490 题增强子集的效果拆解）"""
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
ret = json.load(open(r'output/enhanced_query_result.json', encoding='utf-8'))
assert len(v1) == len(v2) == len(enh)

aug_flag = [any(m.get('added_by') == 'source_coverage' for m in x['retrieval_list']) for x in ret]

agg = defaultdict(lambda: {'v1': 0, 'v2': 0, 'e': 0, 'e_r': 0, 'n': 0})
for a, b, c in zip(v1, v2, enh):
    t = a['question_type']
    agg[t]['n'] += 1
    agg[t]['v1'] += strict(a['model_answer'], a['gold_answer'])
    agg[t]['v2'] += strict(b['model_answer'], b['gold_answer'])
    agg[t]['e'] += strict(c['model_answer'], c['gold_answer'])
    agg[t]['e_r'] += strict(rule(c['model_answer']), c['gold_answer'])

print(f"{'题型':<20}{'v1':>7}{'v2':>7}{'增强':>7}{'增强+规则':>9}{'n':>6}")
T = {'v1': 0, 'v2': 0, 'e': 0, 'e_r': 0, 'n': 0}
for t, v in sorted(agg.items()):
    print(f"{t:<20}{v['v1']/v['n']*100:>6.1f}%{v['v2']/v['n']*100:>6.1f}%{v['e']/v['n']*100:>6.1f}%{v['e_r']/v['n']*100:>8.1f}%{v['n']:>6}")
    for k in T: T[k] += v[k]
print(f"{'总体':<20}{T['v1']/T['n']*100:>6.1f}%{T['v2']/T['n']*100:>6.1f}%{T['e']/T['n']*100:>6.1f}%{T['e_r']/T['n']*100:>8.1f}%{T['n']:>6}")

# 增强子集（490 题）的效果
print(f"\n=== 增强子集（490 题）v2 → 增强 ===")
sub = [(a, b, c) for a, b, c, f in zip(v1, v2, enh, aug_flag) if f]
d2 = sum(strict(b['model_answer'], b['gold_answer']) for a, b, c in sub)
de = sum(strict(c['model_answer'], c['gold_answer']) for a, b, c in sub)
n = len(sub)
print(f"  v2: {d2}/{n} = {d2/n*100:.1f}%  增强: {de}/{n} = {de/n*100:.1f}%  Δ={(de-d2)/n*100:+.1f}pp")
# 分题型
by_t = defaultdict(lambda: {'v2': 0, 'e': 0, 'n': 0})
for a, b, c in sub:
    t = a['question_type']
    by_t[t]['n'] += 1
    by_t[t]['v2'] += strict(b['model_answer'], b['gold_answer'])
    by_t[t]['e'] += strict(c['model_answer'], c['gold_answer'])
for t, v in sorted(by_t.items()):
    print(f"  {t:<20} v2={v['v2']/v['n']*100:>5.1f}%  增强={v['e']/v['n']*100:>5.1f}%  Δ={(v['e']-v['v2'])/v['n']*100:>+5.1f}  n={v['n']}")

# 非增强子集（对照组）
print(f"\n=== 非增强子集（2066 题，对照组，应基本不变）===")
sub2 = [(a, b, c) for a, b, c, f in zip(v1, v2, enh, aug_flag) if not f]
d2 = sum(strict(b['model_answer'], b['gold_answer']) for a, b, c in sub2)
de = sum(strict(c['model_answer'], c['gold_answer']) for a, b, c in sub2)
n2 = len(sub2)
print(f"  v2: {d2/n2*100:.1f}%  增强: {de/n2*100:.1f}%  Δ={(de-d2)/n2*100:+.1f}pp（应≈0）")
