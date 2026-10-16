# -*- coding: utf-8 -*-
"""[ADAPTED 2026-10-16] 源覆盖增强层（source coverage guarantee）—— 自研改进模块

原理：题面点名的来源（"as reported by X and Y"）在检索结果中缺席时，
从该来源的 chunks 池里 BM25 粗筛 + BGE 重排精筛，补充最相关的 2 段进 context。
不重跑基础检索（纯 post-processing）；不使用 question_type/answer/evidence_list 任何标注。

输入：output/full_query_result.json + storage docstore（与检索器完全同源 chunks）
输出：output/enhanced_query_result.json

用法：python enhance_sources.py [limit]   （limit 为可选的试跑题数）
"""
import sys, io, json, re
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CODE = r'E:\NLP_project\code'
DOCSTORE = CODE + r'\storage\all-MiniLM-L6-v2_chunk256\docstore.json'
IN_FILE = CODE + r'\output\full_query_result.json'
OUT_FILE = CODE + r'\output\enhanced_query_result.json'
RERANKER_PATH = r'E:\NLP_project\models\bge-reranker-v2-m3'

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
MAX_SRC_PER_Q = 3   # 每题最多补 3 个来源
CHUNKS_PER_SRC = 2  # 每个来源补 2 段

# ---------- [1] 载入 chunks 池 ----------
print('[1/5] 载入 docstore chunks ...')
raw = json.load(open(DOCSTORE, encoding='utf-8'))
data = raw.get('docstore/data', raw)
chunks = []
for _, wrapper in data.items():
    d = wrapper.get('__data__', wrapper)
    text = d.get('text', '')
    md = d.get('metadata', {}) or {}
    src = str(md.get('source', '')).strip()
    if text and src:
        chunks.append({
            'text': text, 'source': src,
            'title': md.get('title'), 'published_at': md.get('published_at'),
        })
by_source = defaultdict(list)
for c in chunks:
    by_source[c['source']].append(c)
sources = sorted(by_source.keys())
print(f'  chunks: {len(chunks)}, sources: {len(sources)}')

# ---------- [2] 来源匹配键（含"|"长名的首段简称） ----------
def keys_of(s):
    ks = set()
    if len(s) >= 4:
        ks.add(s)
    first = s.split('|')[0].strip()
    if len(first) >= 4:
        ks.add(first)
    return ks

key_to_source = {}
for s in sources:
    for k in keys_of(s):
        key_to_source.setdefault(k, s)
# 长键优先 + 词边界（避免 "The Age" 命中 "The Agency"）
key_pat = [
    (k, re.compile(r'(?<![A-Za-z0-9])' + re.escape(k) + r'(?![A-Za-z0-9])'))
    for k in sorted(key_to_source.keys(), key=len, reverse=True)
]
print(f'  匹配键: {len(key_pat)} 个')

# ---------- [3] 扫描检索结果，找缺席来源 ----------
print('[2/5] 扫描题面点名来源与覆盖情况 ...')
results = json.load(open(IN_FILE, encoding='utf-8'))
if LIMIT:
    results = results[:LIMIT]

need = {}  # q_index -> [missing sources]
stats = {'queries': len(results), 'queries_with_named_src': 0,
         'queries_augmented': 0, 'sources_missing': 0}
for i, item in enumerate(results):
    present = set(x.get('source') for x in item['retrieval_list'])
    wanted = []
    for k, pat in key_pat:
        if pat.search(item['query']):
            s = key_to_source[k]
            if s not in wanted:
                wanted.append(s)
    if wanted:
        stats['queries_with_named_src'] += 1
    missing = [s for s in wanted if s not in present]
    if missing:
        need[i] = missing[:MAX_SRC_PER_Q]
        stats['queries_augmented'] += 1
        stats['sources_missing'] += len(missing)
print(f"  点名来源的题: {stats['queries_with_named_src']}/{stats['queries']}; 需补: {stats['queries_augmented']} 题, 缺席来源 {stats['sources_missing']} 个")

# ---------- [4] BM25 粗筛候选 ----------
print('[3/5] BM25 粗筛缺席来源候选 ...')
from rank_bm25 import BM25Okapi

def tok(s):
    return re.findall(r'[a-z0-9]+', s.lower())

bm25_cache = {}
cand = {}
for i, miss in need.items():
    qt = tok(results[i]['query'])
    for s in miss:
        if s not in bm25_cache:
            pool = by_source[s]
            bm25_cache[s] = (pool, BM25Okapi([tok(c['text']) for c in pool]))
        pool, bm25 = bm25_cache[s]
        scores = bm25.get_scores(qt)
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:6]
        cand[(i, s)] = [pool[j] for j in top_idx]
print(f'  候选组: {len(cand)}')

# ---------- [5] BGE 重排 + 写回 ----------
print('[4/5] BGE 重排精筛 ...')
from FlagEmbedding import FlagReranker
reranker = FlagReranker(RERANKER_PATH, use_fp16=True)

added = 0
for i, miss in need.items():
    item = results[i]
    q = item['query']
    for s in miss:
        cands = cand[(i, s)]
        scores = reranker.compute_score([(q, c['text']) for c in cands])
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[:CHUNKS_PER_SRC]
        for sc, c in ranked:
            item['retrieval_list'].append({
                'text': c['text'],
                'title': c.get('title'),
                'source': c.get('source'),
                'published_at': c.get('published_at'),
                'score': float(sc),
                'added_by': 'source_coverage',
            })
            added += 1
stats['chunks_added'] = added

print('[5/5] 保存 ...')
json.dump(results, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'  新增 chunks: {added}')
print('  stats:', stats)
print(f'  saved: {OUT_FILE}')
