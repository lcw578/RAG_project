# -*- coding: utf-8 -*-
"""对照实验：gold 文本在 prompt 中的出现率 —— 证据命中组 vs 证据未命中组
   若"出现"是自然共现（答案随证据而来），未命中组应显著更低；若是 leaked 则两组应相近。"""
import sys, io, json
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def norm(s):
    return s.replace(' ', '').replace('\n', '').lower()

pv2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
ret = json.load(open(r'output/full_query_result.json', encoding='utf-8'))

def covered(item):
    texts = [norm(m['text']) for m in item['retrieval_list']]
    tot = 0; cov = 0
    for f in item['gold_list']:
        tot += 1
        if any(norm(f['fact']) in t for t in texts):
            cov += 1
    return cov, tot

# 只看长答案（>=8 字符），避免 "Yes"/"no" 之类的自然语言干扰
grp = {'hit(cov>=1)': [0, 0], 'miss(cov=0)': [0, 0]}   # [gold_in_prompt, total]
for r, x in zip(ret, pv2):
    g = (x.get('gold_answer') or '').strip()
    if len(g) < 8:
        continue
    cov, tot = covered(r)
    key = 'hit(cov>=1)' if cov >= 1 else 'miss(cov=0)'
    grp[key][1] += 1
    if g.lower() in x['prompt'].lower():
        grp[key][0] += 1

print("=== 长 gold（>=8字符）出现在 prompt 中的比例 ===")
for k, (a, b) in grp.items():
    print(f"  {k:<14} {a}/{b} = {a/b*100:.1f}%")

print("\n=== 解读 ===")
print("若未命中组显著更低 → gold 文本的出现是'随证据捞上来的'（RAG 正常现象），非注入泄漏")
