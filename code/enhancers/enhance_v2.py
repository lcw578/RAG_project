# -*- coding: utf-8 -*-
"""[ADAPTED 2026-10-16] 检索后处理 #3+#4：来源补强 v2 + 日期匹配

① 来源补强 v2：题面点名的每个来源，保证 context 中 ≥3 段（当前 <3 的用 BM25+重排补齐）
② 日期匹配：题面点名的显式日期（October 7, 2023），若语料有该日期的文章、但 context 缺席，
   则从该日期的 chunks 里 BM25+重排补 2 段
不使用 question_type/answer/evidence_list 任何标注；纯题面信号 + 语料元数据。

输入：output/enhanced_query_result.json（源覆盖后、未日期标注）
输出：output/final_query_result_pre.json（随后交 temporal_annotate.py 统一加日期前缀）
"""
import sys, io, json, re
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CODE = r'E:\NLP_project\code'
DOCSTORE = CODE + r'\storage\all-MiniLM-L6-v2_chunk256\docstore.json'
IN_FILE = CODE + r'\output\enhanced_query_result.json'
OUT_FILE = CODE + r'\output\final_query_result_pre.json'
RERANKER_PATH = r'E:\NLP_project\models\bge-reranker-v2-m3'

SRC_TARGET = 3      # 点名来源的目标段数
SRC_ADD_CAP = 2     # 每个来源最多补 2 段
DATE_ADD = 2        # 每个缺席日期补 2 段
TOTAL_ADD_CAP = 6   # 每题总补充上限
TOTAL_CHUNK_CAP = 18

MONTHS = {'january':1,'jan':1,'february':2,'feb':2,'march':3,'mar':3,'april':4,'apr':4,
          'may':5,'june':6,'jun':6,'july':7,'jul':7,'august':8,'aug':8,'september':9,'sept':9,'sep':9,
          'october':10,'oct':10,'november':11,'nov':11,'december':12,'dec':12}
DATE_RE = re.compile(r'\b(' + '|'.join(sorted(MONTHS, key=len, reverse=True)) + r')\.?\s+(\d{1,2}),?\s+(\d{4})\b', re.I)

# ---------- [1] chunks 池 ----------
print('[1/6] 载入 docstore ...')
raw = json.load(open(DOCSTORE, encoding='utf-8'))
data = raw.get('docstore/data', raw)
chunks = []
for _, w in data.items():
    d = w.get('__data__', w)
    text = d.get('text', '')
    md = d.get('metadata', {}) or {}
    src = str(md.get('source', '')).strip()
    pa = str(md.get('published_at') or '')[:10]
    if text and src:
        chunks.append({'text': text, 'source': src, 'date': pa,
                       'title': md.get('title'), 'published_at': md.get('published_at')})
by_source = defaultdict(list)
by_date = defaultdict(list)
date_set = set()
for c in chunks:
    by_source[c['source']].append(c)
    if len(c['date']) == 10:
        by_date[c['date']].append(c)
        date_set.add(c['date'])
print(f'  chunks: {len(chunks)}, sources: {len(by_source)}, dates: {len(by_date)}')

# ---------- [2] 来源匹配键 ----------
def keys_of(s):
    ks = set()
    if len(s) >= 4: ks.add(s)
    first = s.split('|')[0].strip()
    if len(first) >= 4: ks.add(first)
    return ks
key_to_source = {}
for s in by_source:
    for k in keys_of(s):
        key_to_source.setdefault(k, s)
key_pat = [(k, re.compile(r'(?<![A-Za-z0-9])' + re.escape(k) + r'(?![A-Za-z0-9])'))
           for k in sorted(key_to_source, key=len, reverse=True)]

# ---------- [3] 扫描 ----------
print('[2/6] 扫描检索结果 ...')
results = json.load(open(IN_FILE, encoding='utf-8'))
need_src = {}   # i -> {source: add_k}
need_date = {}  # i -> [dates]
TOTAL_ADD_CAP_local = TOTAL_ADD_CAP
for i, item in enumerate(results):
    q = item['query']
    cnt = Counter(m.get('source') for m in item['retrieval_list'])
    wanted = []
    for k, pat in key_pat:
        if pat.search(q):
            s = key_to_source[k]
            if s not in wanted:
                wanted.append(s)
    topup = {}
    for s in wanted:
        c = cnt.get(s, 0)
        if c < SRC_TARGET:
            topup[s] = min(SRC_TARGET - c, SRC_ADD_CAP)
    if topup:
        need_src[i] = topup
    # 日期
    ctx_dates = set(str(m.get('published_at') or '')[:10] for m in item['retrieval_list'])
    qdates = []
    for m in DATE_RE.finditer(q):
        mon = MONTHS[m.group(1).lower()]
        d = f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
        if d in date_set and d not in ctx_dates and d not in qdates:
            qdates.append(d)
    if qdates:
        need_date[i] = qdates

print(f'  来源补强: {len(need_src)} 题; 日期匹配: {len(need_date)} 题')

# ---------- [4] BM25 候选 ----------
print('[3/6] BM25 粗筛 ...')
from rank_bm25 import BM25Okapi
def tok(s):
    return re.findall(r'[a-z0-9]+', s.lower())
bm25_cache = {}
def get_cand(pool, q, topk=6):
    key = id(pool)
    if key not in bm25_cache:
        bm25_cache[key] = BM25Okapi([tok(c['text']) for c in pool])
    bm25 = bm25_cache[key]
    scores = bm25.get_scores(tok(q))
    idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:topk]
    return [pool[j] for j in idx]

cand = {}
for i, topup in need_src.items():
    q = results[i]['query']
    for s, k in topup.items():
        cand[('S', i, s)] = get_cand(by_source[s], q)
for i, qds in need_date.items():
    q = results[i]['query']
    for d in qds:
        cand[('D', i, d)] = get_cand(by_date[d], q)
print(f'  候选组: {len(cand)}')

# ---------- [5] 重排 + 写回 ----------
print('[4/6] BGE 重排 ...')
from FlagEmbedding import FlagReranker
reranker = FlagReranker(RERANKER_PATH, use_fp16=True)

added_total = 0
for i in range(len(results)):
    item = results[i]
    if i not in need_src and i not in need_date:
        continue
    q = item['query']
    present_texts = set(m['text'] for m in item['retrieval_list'])
    budget = TOTAL_ADD_CAP
    new = []
    # 来源补强
    for s, k in (need_src.get(i) or {}).items():
        if budget <= 0: break
        cs = [c for c in cand[('S', i, s)] if c['text'] not in present_texts]
        if not cs: continue
        scores = reranker.compute_score([(q, c['text']) for c in cs])
        if isinstance(scores, float): scores = [scores]
        ranked = sorted(zip(scores, cs), key=lambda x: -x[0])[:k]
        for sc, c in ranked:
            if budget <= 0: break
            new.append(('source_topup', sc, c)); budget -= 1
    # 日期匹配
    for d in (need_date.get(i) or []):
        if budget <= 0: break
        cs = [c for c in cand[('D', i, d)] if c['text'] not in present_texts and c['text'] not in [x[2]['text'] for x in new]]
        if not cs: continue
        scores = reranker.compute_score([(q, c['text']) for c in cs])
        if isinstance(scores, float): scores = [scores]
        ranked = sorted(zip(scores, cs), key=lambda x: -x[0])[:DATE_ADD]
        for sc, c in ranked:
            if budget <= 0: break
            new.append(('date_match', sc, c)); budget -= 1
    room = TOTAL_CHUNK_CAP - len(item['retrieval_list'])
    for tag, sc, c in new[:max(0, room)]:
        item['retrieval_list'].append({
            'text': c['text'], 'title': c.get('title'), 'source': c.get('source'),
            'published_at': c.get('published_at'), 'score': float(sc), 'added_by': tag,
        })
        added_total += 1

print(f'[5/6] 新增 chunks: {added_total}')
json.dump(results, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
stats = {
    'queries': len(results),
    'src_topup_queries': len(need_src),
    'date_match_queries': len(need_date),
    'chunks_added': added_total,
}
print('[6/6] stats:', stats)
print('saved:', OUT_FILE)
