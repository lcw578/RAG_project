# -*- coding: utf-8 -*-
"""v2 质量检查：翻转题的行为分析 + 实样抽查"""
import sys, io, json
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def strict(pred, gold):
    return bool(set(pred.lower().split()) & set(gold.lower().split()))

def is_ii(s):
    return 'insufficient' in s.lower()

v1 = json.load(open(r'qa_output/full_qwen.json', encoding='utf-8'))
v2 = json.load(open(r'qa_output/pv2_full.json', encoding='utf-8'))
ret = json.load(open(r'output/full_query_result.json', encoding='utf-8'))

# ============ 1) comparison: II→作答 的翻转质量 ============
flips = [(r, a, b) for r, a, b in zip(ret, v1, v2)
         if r['question_type'] == 'comparison_query' and is_ii(a['model_answer']) and not is_ii(b['model_answer'])]
print(f"=== comparison: v1 答 II → v2 给出实答 的翻转: {len(flips)} 题 ===")
correct = sum(1 for _, _, b in flips if strict(b['model_answer'], b['gold_answer']))
print(f"其中答对: {correct} ({correct/len(flips)*100:.1f}%)")
ans_dist = Counter(b['model_answer'].strip().lower()[:10] for _, _, b in flips)
gold_dist = Counter(b['gold_answer'].strip().lower()[:10] for _, _, b in flips)
print(f"v2 答案分布: {dict(ans_dist.most_common(6))}")
print(f"gold 分布:   {dict(gold_dist.most_common(6))}")
# 若模型只是无脑答 Yes 会怎样？基线率：
yes_rate = sum(1 for _, _, b in flips if 'yes' in b['model_answer'].lower()) / len(flips)
print(f"v2 答 'yes' 比例: {yes_rate*100:.1f}%  （无脑全答 yes 的理论命中率 = gold yes 占比 {gold_dist.get('yes',0)/len(flips)*100:.1f}%）")

# ============ 2) comparison: 反向翻转（v1 对 → v2 错）= 伤害检查 ============
hurt = [(r, a, b) for r, a, b in zip(ret, v1, v2)
        if r['question_type'] == 'comparison_query' and strict(a['model_answer'], a['gold_answer']) and not strict(b['model_answer'], b['gold_answer'])]
print(f"\n=== comparison: v1 对 → v2 错 的伤害: {len(hurt)} 题 ===")
for r, a, b in hurt[:5]:
    print(f"  gold={r['answer']!r} v1={a['model_answer'][:30]!r} v2={b['model_answer'][:40]!r}")

# ============ 3) null: v2 仍在乱答的（非 II 且错）============
null_bad = [(r, b) for r, a, b in zip(ret, v1, v2)
            if r['question_type'] == 'null_query' and not is_ii(b['model_answer']) and len(b['model_answer'].strip()) > 1]
print(f"\n=== null: v2 仍在实答（应拒答）: {len(null_bad)} 题 ===")
for r, b in null_bad[:5]:
    print(f"  v2={b['model_answer'][:50]!r}")

# ============ 4) inference: 回退例 ============
inf_hurt = [(r, a, b) for r, a, b in zip(ret, v1, v2)
            if r['question_type'] == 'inference_query' and strict(a['model_answer'], a['gold_answer']) and not strict(b['model_answer'], b['gold_answer'])]
print(f"\n=== inference: v1 对 → v2 错: {len(inf_hurt)} 题，抽样：")
for r, a, b in inf_hurt[:4]:
    print(f"  gold={r['answer'][:40]!r} v1={a['model_answer'][:30]!r} v2={b['model_answer'][:60]!r}")

# ============ 5) 实样抽查：翻转且答对的 comparison 3 例（看证据是否支持）============
print(f"\n=== 实样抽查：翻转且答对的 comparison（证据支持性）===")
shown = 0
for r, a, b in flips:
    if strict(b['model_answer'], b['gold_answer']) and shown < 3:
        shown += 1
        print(f"\n--- 样例 {shown} ---")
        print(f"Q: {r['query'][:200]}")
        print(f"gold={r['answer']!r}  v2 答={b['model_answer']!r}")
        for m in r['retrieval_list'][:2]:
            print(f"  证据[{m.get('rerank_rank')}]: {m['text'][:160]}")
