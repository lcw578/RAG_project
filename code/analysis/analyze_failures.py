# -*- coding: utf-8 -*-
"""失败归因深挖：comparison/temporal 失败者的证据覆盖度 + null 失败回答分布"""
import sys, io, json
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def norm(s):
    return s.replace(' ', '').replace('\n', '').lower()

def fact_coverage(item):
    texts = [norm(m['text']) for m in item['retrieval_list']]
    covered = 0
    for f in item['gold_list']:
        fn = norm(f['fact'])
        if any(fn in t for t in texts):
            covered += 1
    return covered, len(item['gold_list'])

ret = json.load(open(r'output/full_query_result.json', encoding='utf-8'))
qa = json.load(open(r'qa_output/full_qwen.json', encoding='utf-8'))
qa_by = {x['query']: x for x in qa}

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

comp = Counter()
temporal = Counter()
for item in ret:
    t = item['question_type']
    if t not in ('comparison_query', 'temporal_query'):
        continue
    pred = qa_by[item['query']]['model_answer']
    if strict(pred, item['answer']):
        continue
    cov, tot = fact_coverage(item)
    bucket = f"{cov}/{tot} facts covered"
    if t == 'comparison_query':
        comp[bucket] += 1
    else:
        temporal[bucket] += 1

print('=== comparison 失败者的证据覆盖度 ===')
for k, v in sorted(comp.items()):
    print(f"  {k}: {v}")
print('=== temporal 失败者的证据覆盖度 ===')
for k, v in sorted(temporal.items()):
    print(f"  {k}: {v}")

null_fails = Counter()
null_answers = []
for item in ret:
    if item['question_type'] != 'null_query':
        continue
    pred = qa_by[item['query']]['model_answer']
    if not strict(pred, item['answer']):
        if len(pred) <= 2:
            null_fails['垃圾短答案(<=2字符)'] += 1
        elif 'insufficient' in pred.lower():
            null_fails['答了但被判错(含Insufficient)'] += 1
        else:
            null_fails['一本正经地答错'] += 1
        if len(null_answers) < 10:
            null_answers.append(pred[:70])
print('=== null 失败回答分布 ===')
tot = sum(null_fails.values())
print(f"  总失败: {tot} / 301")
for k, v in null_fails.items():
    print(f"  {k}: {v} ({v/tot*100:.0f}%)")
for a in null_answers[:10]:
    print(f"    {a!r}")

comp_all = Counter()
for item in ret:
    if item['question_type'] != 'comparison_query':
        continue
    pred = qa_by[item['query']]['model_answer']
    ok = strict(pred, item['answer'])
    cov, tot = fact_coverage(item)
    comp_all[('答对' if ok else '答错', f"{cov}/{tot}")] += 1
print('=== comparison 全部题：覆盖度 × 对错 ===')
for k, v in sorted(comp_all.items()):
    print(f"  {k}: {v}")
