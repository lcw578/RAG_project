# -*- coding: utf-8 -*-
"""v3 剩余失败解剖：按 检索缺口/题型结构/答案分布 找出剩余优化点"""
import sys, io, json, re
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

def rule(ans):
    return "Insufficient Information" if len(ans.strip()) <= 1 else ans

def norm(s):
    return s.replace(' ', '').replace('\n', '').lower()

def coverage(item):
    texts = [norm(m['text']) for m in item['retrieval_list']]
    tot = cov = 0
    for f in item['gold_list']:
        tot += 1
        if any(norm(f['fact']) in t for t in texts):
            cov += 1
    return cov, tot

v3 = json.load(open(r'qa_output/v3_full.json', encoding='utf-8'))
ret = json.load(open(r'output/enhanced_temporal_query_result.json', encoding='utf-8'))

MONTHS = ['january','february','march','april','may','june','july','august','september','october','november','december']
DATE_RE = re.compile(r'\b(' + '|'.join(MONTHS) + r')\s+\d{1,2}(,?\s*\d{4})?', re.I)

stats = defaultdict(Counter)
fail_examples = defaultdict(list)
yesno = defaultdict(Counter)

for a, r in zip(v3, ret):
    t = a['question_type']
    ok = strict(rule(a['model_answer']), a['gold_answer'])
    if ok:
        continue
    cov, tot = coverage(r)
    bucket = 'A.检索没到(0facts)' if cov == 0 else ('B.只到一部分' if cov < tot else 'C.证据齐全仍错')
    stats[t][bucket] += 1
    if len(fail_examples[bucket]) < 3:
        fail_examples[bucket].append((t, a['query'][:100], r['answer'][:30], rule(a['model_answer'])[:50]))
    if t == 'comparison_query':
        g = r['answer'].strip().lower()
        pred = rule(a['model_answer']).strip().lower()
        gk = 'yes' if g in ('yes','true','agree','similar') else ('no' if g in ('no','false') else 'other')
        pk = 'yes' if pred == 'yes' else ('no' if pred == 'no' else ('II' if 'insufficient' in pred else 'other'))
        yesno[f'gold={gk}'][pk] += 1

print('=== v3 剩余失败按题型 × 检索缺口 ===')
total = 0
for t in ['comparison_query', 'temporal_query', 'inference_query', 'null_query']:
    s = stats[t]
    n = sum(s.values())
    total += n
    print(f"  {t:<18} 失败 {n:>3}: " + '; '.join(f"{k}={v}" for k, v in sorted(s.items())))
print(f"  合计失败 {total}")

print('\n=== comparison 失败者的金标 vs 模型答案分布 ===')
for gk in sorted(yesno):
    print(f"  {gk}: " + '; '.join(f"模型答[{pk}]={v}" for pk, v in yesno[gk].most_common()))

print('\n=== temporal 失败者：题面是否含显式日期 ===')
tf_d = tf_nd = 0
tt_d = tt_nd = 0
for a, r in zip(v3, ret):
    if a['question_type'] != 'temporal_query':
        continue
    ok = strict(rule(a['model_answer']), a['gold_answer'])
    has_date = bool(DATE_RE.search(r['query']))
    if ok:
        if has_date: tt_d += 1
        else: tt_nd += 1
    else:
        if has_date: tf_d += 1
        else: tf_nd += 1
print(f"  对: 含日期 {tt_d}, 不含 {tt_nd}（含日期题正确率 {tt_d/(tt_d+tf_d)*100:.1f}%）")
print(f"  错: 含日期 {tf_d}, 不含 {tf_nd}（不含日期题正确率 {tt_nd/(tt_nd+tf_nd)*100:.1f}%）")

print('\n=== 失败样例（每类 3 条）===')
for b in ['A.检索没到(0facts)', 'B.只到一部分', 'C.证据齐全仍错']:
    print(f"--- {b} ---")
    for t, q, g, p in fail_examples[b]:
        print(f"  [{t[:10]}] Q:{q}...")
        print(f"      gold={g!r}  pred={p!r}")
