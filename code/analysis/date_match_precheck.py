# -*- coding: utf-8 -*-
"""预检：题面里的显式日期 × 语料覆盖 —— 日期匹配检索值不值得做"""
import sys, io, json, re
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MONTHS = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
DATE_RE = re.compile(r'\b(' + '|'.join(MONTHS) + r')\s+(\d{1,2}),?\s+(\d{4})\b', re.I)

corpus = json.load(open(r'dataset/corpus.json', encoding='utf-8'))
ret = json.load(open(r'output/enhanced_temporal_query_result.json', encoding='utf-8'))

# 语料日期集合
corpus_dates = set()
for d in corpus:
    pa = str(d.get('published_at') or '')[:10]
    if len(pa) == 10:
        corpus_dates.add(pa)
print(f"语料覆盖的日期数: {len(corpus_dates)}")

def dates_in_query(q):
    out = []
    for m in DATE_RE.finditer(q):
        mon, day, year = m.group(1).lower(), int(m.group(2)), m.group(3)
        out.append(f"{year}-{MONTHS.get(mon,0):02d}-{day:02d}")
    return out

stat = Counter()
q_with_date = 0
date_missing_corpus = 0
date_missing_in_ctx = 0
total_dates = 0
for x in ret:
    ds = dates_in_query(x['query'])
    if not ds:
        continue
    q_with_date += 1
    ctx_dates = set(str(m.get('published_at') or '')[:10] for m in x['retrieval_list'])
    for d in ds:
        total_dates += 1
        if d not in corpus_dates:
            date_missing_corpus += 1
        elif d not in ctx_dates:
            date_missing_in_ctx += 1

print(f"题面含显式日期的题: {q_with_date} / 2556")
print(f"引用的日期总数: {total_dates}")
print(f"  · 语料里没有该日期的文章: {date_missing_corpus}")
print(f"  · 语料里有、但没被检索到 context 里: {date_missing_in_ctx}  ← 日期匹配能补的量")
# 按题型分
by_t = Counter()
for x in ret:
    ds = dates_in_query(x['query'])
    if not ds:
        continue
    ctx_dates = set(str(m.get('published_at') or '')[:10] for m in x['retrieval_list'])
    miss = sum(1 for d in ds if d in corpus_dates and d not in ctx_dates)
    if miss:
        by_t[x['question_type']] += 1
print(f"有'可补日期'的题按题型: {dict(by_t)}")
