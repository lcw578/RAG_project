# -*- coding: utf-8 -*-
"""qa_llama_v2.py — prompt 变体实验版 [ADAPTED 2026-10-16]

与官方 qa_llama.py 的差异仅一处：build_prompt 的前缀指令（针对失败归因发现的两大病症）：
  1. comparison 题“证据齐全却答 Insufficient Information”（192 例）→ 引导逐源核对、明确 Yes 判定条件
  2. null 题输出单字母垃圾答案（85 例）→ 明确禁止猜测与单字母输出
其余逻辑（端点、逐题新建连接、思考开关、输出格式）与 qa_llama.py 完全一致。
用法：python qa_llama_v2.py --input <retrieval.json> --output <out.json> [--thinking off]
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM answers from retrieval-result JSON (prompt variant v2)."
    )
    parser.add_argument("--input", default="output/hybrid_query_result.json")
    parser.add_argument("--output", default="qa_output/hybrid_qwen_v2.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    # [ADAPTED] 思考模式开关（原版 --disable_thinking 为 store_true 且 default=True，永远禁用）
    parser.add_argument("--thinking", choices=["off", "on"], default="off")
    return parser.parse_args()


def build_client() -> OpenAI:
    # [ADAPTED] 指向本机 llama-server（OpenAI 兼容端点）
    api_key = os.getenv("LOCAL_LLM_API_KEY", "sk-local")
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:18080/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def normalize_input_data(doc_data: Any) -> List[Dict[str, Any]]:
    if isinstance(doc_data, list):
        return doc_data
    if isinstance(doc_data, dict):
        return [doc_data]
    raise ValueError("Input JSON must be either a list or a single dict.")


def build_prompt(item: Dict[str, Any]) -> str:
    # [ADAPTED] v2 前缀：comparison 逐源核对 + null 禁猜禁单字母（原官方前缀见 qa_llama.py）
    prefix = (
        "Below is a question followed by some context from different sources. "
        "Please answer the question based on the context. The answer to the "
        "question is a word, entity, or Yes/No. "
        "If the question asks whether two or more sources report something, check "
        "what EACH of those sources says in the context first, and answer Yes only "
        "when the context supports the claim for every source mentioned. "
        "If the provided information is insufficient to answer the question, "
        "respond 'Insufficient Information'. Never guess, and never answer with a "
        "single letter or a fragment. "
        "Answer directly without explanation."
    )

    retrieval_list = item.get("retrieval_list", [])
    context = "\n\n--------------\n\n".join(
        evidence.get("text", "")
        for evidence in retrieval_list
    )

    query = item["query"]

    prompt = (
        f"{prefix}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n\n{context}"
    )

    return prompt


def generate_answer(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    disable_thinking: bool = True,
) -> str:
    extra_body = {}

    if disable_thinking:
        # [ADAPTED] llama.cpp 通过 chat_template_kwargs 传思考开关
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_new_tokens,
        extra_body=extra_body if extra_body else None,
    )

    return response.choices[0].message.content.strip()


def main() -> None:
    load_dotenv()

    args = parse_args()
    model_name = os.getenv("LOCAL_LLM_MODEL", "local")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Retrieval result file was not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        raw_doc_data = json.load(file)

    doc_data = normalize_input_data(raw_doc_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Local LLM: {model_name}")
    print(f"Number of questions: {len(doc_data)}")

    save_list = []

    for index, item in enumerate(tqdm(doc_data, desc="Generating answers"), start=1):
        prompt = build_prompt(item)

        print(f"Generating answer for query {index}/{len(doc_data)}...")

        # [ADAPTED] 每题新建连接：实测复用持久连接时 llama-server 会隔次返回 404
        client = build_client()

        response = generate_answer(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=(args.thinking == "off"),
        )
        client.close()

        save_list.append(
            {
                "query": item["query"],
                "prompt": prompt,
                "model_answer": response,
                "gold_answer": item.get("answer"),
                "question_type": item.get("question_type"),
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(save_list, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(save_list)} model answers to: {output_path}")


if __name__ == "__main__":
    main()
