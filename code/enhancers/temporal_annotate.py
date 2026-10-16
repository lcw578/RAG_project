# -*- coding: utf-8 -*-
"""[ADAPTED 2026-10-16] temporal 时间感知 —— 检索后处理 #2：日期标注

在每段证据文本前加 "[Published: YYYY-MM-DD]" 前缀（数据来自语料 published_at 元数据），
让模型在跨文档时间比较题里"看得见"时间。不修改任何标注字段。

输入：output/enhanced_query_result.json
输出：output/enhanced_temporal_query_result.json
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CODE = r'E:\NLP_project\code'
IN_FILE = sys.argv[1] if len(sys.argv) > 1 else CODE + r'\output\enhanced_query_result.json'
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else CODE + r'\output\enhanced_temporal_query_result.json'

data = json.load(open(IN_FILE, encoding='utf-8'))
n_chunks = n_annotated = n_missing = 0
for item in data:
    for m in item['retrieval_list']:
        n_chunks += 1
        pa = str(m.get('published_at') or '').strip()
        if pa and len(pa) >= 10:
            date_only = pa[:10]  # '2023-10-07T...' -> '2023-10-07'
            m['text'] = f"[Published: {date_only}] " + m['text']
            n_annotated += 1
        else:
            n_missing += 1

json.dump(data, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"chunks 总数: {n_chunks}; 已标注: {n_annotated}; 缺日期: {n_missing}")
print(f"saved: {OUT_FILE}")

# 抽样展示
x = next(x for x in data if x['question_type'] == 'temporal_query')
print('\n时间题样例（前 3 段证据开头）:')
for m in x['retrieval_list'][:3]:
    print('  ', m['text'][:110])
