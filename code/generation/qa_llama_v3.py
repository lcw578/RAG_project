# -*- coding: utf-8 -*-
"""qa_llama_v3.py — prompt 变体 v3 [ADAPTED 2026-10-16]

= v2 prompt + temporal 时间引导（条件式："当问题涉及事件顺序/随时间变化时…使用括号里的发布日期"）。
其余与 qa_llama_v2.py 完全一致。
用法：python qa_llama_v3.py --input <annotated_retrieval.json> --output <out.json>
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
    parser = argparse.ArgumentParser(description="Generate LLM answers (prompt variant v3).")
    parser.add_argument("--input", default="output/hybrid_query_result.json")
    parser.add_argument("--output", default="qa_output/hybrid_qwen_v3.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", choices=["off", "on"], default="off")
    return parser.parse_args()


def build_client() -> OpenAI:
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
    # [ADAPTED] v3 前缀：v2 + 时间引导
    prefix = (
        "Below is a question followed by some context from different sources. "
        "Please answer the question based on the context. The answer to the "
        "question is a word, entity, or Yes/No. "
        "If the question asks whether two or more sources report something, check "
        "what EACH of those sources says in the context first, and answer Yes only "
        "when the context supports the claim for every source mentioned. "
        "When the question involves the order of events, or how something changed "
        "over time, use the publication dates shown in brackets to establish the "
        "timeline before answering. "
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

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(save_list)} model answers to: {output_path}")


if __name__ == "__main__":
    main()
