"""评估脚本：对比基线 RAG 与纠错 RAG 的检索质量和回答质量。

需要真实的 API 密钥（OpenAI 必填；Tavily 可选，用于验证网络搜索兜底）以及可连接的 Qdrant。

用法示例：
    uv run python evaluate.py --llm-api-key sk-xxx --tavily-api-key tvly-xxx \
        --qdrant-url http://localhost:6333 --limit 5

也可以直接设置环境变量 LLM_API_KEY / TAVILY_API_KEY / QDRANT_URL / QDRANT_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv 未安装时静默跳过
    pass

from langchain_openai import ChatOpenAI  # noqa: E402

from corrective_rag.core import CorrectiveRAG, load_local_path  # noqa: E402

JUDGE_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash-vision-exp")


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def judge(
    openai_api_key: str,
    prompt: str,
    base_url: str = "",
    max_retries: int = 3,
) -> dict:
    """调用默认大语言模型（deepseek-v4-flash-vision-exp）做二元打分，返回解析后的 JSON。"""
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=openai_api_key,
        base_url=base_url or None,
        temperature=0,
        max_tokens=64,
    )
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            parsed = _extract_json(response.content)
            if parsed:
                return parsed
        except Exception:
            if attempt == max_retries - 1:
                return {}
    return {}


def judge_relevance(openai_api_key: str, question: str, chunk: str, base_url: str = "") -> bool:
    prompt = f"""判断下面的检索片段是否与问题相关。
规则：只输出 JSON，格式为 {{"score": "yes"}} 或 {{"score": "no"}}，不要输出其他内容。
问题：{question}
片段：{chunk[:1500]}"""
    return judge(openai_api_key, prompt, base_url=base_url).get("score") == "yes"


def judge_faithfulness(
    openai_api_key: str, question: str, answer: str, context: str, base_url: str = ""
) -> bool:
    prompt = f"""判断回答是否完全由给定的上下文支撑（不允许编造上下文之外的事实）。
规则：只输出 JSON，格式为 {{"score": 1}} 或 {{"score": 0}}。
问题：{question}
回答：{answer}
上下文：{context[:4000]}"""
    result = judge(openai_api_key, prompt, base_url=base_url)
    return result.get("score") in (1, "1", True)


def judge_answer_relevance(
    openai_api_key: str, question: str, answer: str, base_url: str = ""
) -> bool:
    prompt = f"""判断回答是否直接回答了问题（不跑题、不答非所问）。
规则：只输出 JSON，格式为 {{"score": 1}} 或 {{"score": 0}}。
问题：{question}
回答：{answer}"""
    result = judge(openai_api_key, prompt, base_url=base_url)
    return result.get("score") in (1, "1", True)


def judge_answer_correctness(
    openai_api_key: str,
    question: str,
    answer: str,
    reference: str,
    base_url: str = "",
) -> bool:
    prompt = f"""判断回答是否与参考答案表达的事实一致（允许措辞不同，但事实必须相同，不得编造）。
规则：只输出 JSON，格式为 {{"score": 1}} 或 {{"score": 0}}。
问题：{question}
参考答案：{reference}
回答：{answer}"""
    result = judge(openai_api_key, prompt, base_url=base_url)
    return result.get("score") in (1, "1", True)


def judge_unsupported_claims(
    openai_api_key: str, answer: str, context: str, base_url: str = ""
) -> int:
    """让裁判逐句统计回答中不被上下文支撑的事实性断言数量。"""
    prompt = f"""逐句检查回答中的事实性断言，统计有多少个断言不被给定上下文支撑。
规则：只输出 JSON，格式为 {{"unsupported_count": 数字}}。
上下文：{context[:4000]}
回答：{answer}"""
    result = judge(openai_api_key, prompt, base_url=base_url)
    count = result.get("unsupported_count", 0)
    try:
        return max(0, int(count))
    except (TypeError, ValueError):
        return 0


def metrics_for(
    openai_api_key: str,
    question: str,
    documents,
    answer: str,
    top_k: int,
    reference: str = "",
    base_url: str = "",
) -> dict:
    """计算单条回答的指标：检索精度、命中率、忠实度（幻觉率）、正确性、相关性。"""
    docs = list(documents)[:top_k]
    if not docs:
        return {
            "precision@k": 0.0,
            "hit_rate": 0.0,
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "unsupported_claims": 0,
            "correctness": 0.0,
            "answer_relevance": 0.0,
            "context_chunks": 0,
        }

    relevant = [
        judge_relevance(openai_api_key, question, doc.page_content, base_url=base_url)
        for doc in docs
    ]
    context = "\n\n".join(doc.page_content[:800] for doc in docs)
    faithfulness = judge_faithfulness(
        openai_api_key, question, answer, context, base_url=base_url
    )
    correctness = (
        judge_answer_correctness(
            openai_api_key, question, answer, reference, base_url=base_url
        )
        if reference
        else 0.0
    )
    return {
        "precision@k": round(sum(relevant) / len(docs), 4),
        "hit_rate": 1.0 if any(relevant) else 0.0,
        "faithfulness": 1.0 if faithfulness else 0.0,
        "hallucination_rate": 0.0 if faithfulness else 1.0,
        "unsupported_claims": judge_unsupported_claims(
            openai_api_key, answer, context, base_url=base_url
        ),
        "correctness": 1.0 if correctness else 0.0,
        "answer_relevance": 1.0
        if judge_answer_relevance(openai_api_key, question, answer, base_url=base_url)
        else 0.0,
        "context_chunks": len(docs),
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        values = [r[key] for r in rows if r is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "precision@k": mean("precision@k"),
        "hit_rate": mean("hit_rate"),
        "faithfulness": mean("faithfulness"),
        "hallucination_rate": mean("hallucination_rate"),
        "unsupported_claims": mean("unsupported_claims"),
        "correctness": mean("correctness"),
        "answer_relevance": mean("answer_relevance"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="纠正式 RAG 评估脚本")
    parser.add_argument("--corpus", default=str(ROOT / "eval/data"))
    parser.add_argument("--questions", default=str(ROOT / "eval/questions.json"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-v4-flash-vision-exp"))
    parser.add_argument(
        "--embedding-model", default=os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    )
    parser.add_argument(
        "--embedding-api-key", default=os.getenv("EMBEDDING_API_KEY", "")
    )
    parser.add_argument(
        "--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL", "")
    )
    parser.add_argument("--tavily-api-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 个问题（0 表示全部）")
    parser.add_argument("--output", default=str(ROOT / "eval/reports/eval_latest.json"))
    args = parser.parse_args()

    if not args.llm_api_key:
        sys.exit("错误：需要大语言模型 API 密钥（--llm-api-key 或环境变量 LLM_API_KEY）")

    corpus = load_local_path(args.corpus)
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    rag = CorrectiveRAG(
        openai_api_key=args.llm_api_key,
        openai_base_url=args.llm_base_url,
        model=args.model,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        tavily_api_key=args.tavily_api_key,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    print(f"正在入库语料（{len(corpus)} 个文档来源）...")
    rag.ingest(corpus)
    print("入库完成，开始评测。\n")

    results = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{index}/{len(questions)}] {question}")
        row = {
            "id": item.get("id", f"q{index}"),
            "question": question,
            "requires_web": item.get("requires_web", False),
            "baseline": None,
            "corrective": None,
            "corrective_steps": [],
        }
        try:
            baseline = rag.run_baseline(question)
            row["baseline"] = metrics_for(
                args.llm_api_key,
                question,
                baseline["documents"],
                baseline["generation"],
                args.top_k,
                item.get("reference", ""),
                args.llm_base_url,
            )
            row["baseline"]["answer"] = baseline["generation"][:200]
        except Exception as e:  # noqa: BLE001
            print(f"  基线流程出错：{e}")

        try:
            steps, final = rag.run(question)
            row["corrective_steps"] = list(steps.keys())
            row["corrective"] = metrics_for(
                args.llm_api_key,
                question,
                final["documents"],
                final["generation"],
                args.top_k,
                item.get("reference", ""),
                args.llm_base_url,
            )
            row["corrective"]["answer"] = final["generation"][:200]
        except Exception as e:  # noqa: BLE001
            print(f"  纠错流程出错：{e}")

        results.append(row)
        print(f"  基线: {row['baseline']}")
        print(f"  纠错: {row['corrective']}  步骤: {row['corrective_steps']}\n")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "corpus": args.corpus,
            "questions": args.questions,
            "top_k": args.top_k,
            "judge_model": JUDGE_MODEL,
            "tavily_configured": bool(args.tavily_api_key),
        },
        "questions_count": len(results),
        "baseline_aggregate": aggregate([r["baseline"] for r in results]),
        "corrective_aggregate": aggregate([r["corrective"] for r in results]),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 汇总 ===")
    print(f"指标            基线      纠错")
    for key in (
        "precision@k",
        "hit_rate",
        "faithfulness",
        "hallucination_rate",
        "unsupported_claims",
        "correctness",
        "answer_relevance",
    ):
        base = report["baseline_aggregate"][key]
        corr = report["corrective_aggregate"][key]
        print(f"{key:<15}{base:<10.4f}{corr:<10.4f}")

    base_hall = report["baseline_aggregate"]["hallucination_rate"]
    corr_hall = report["corrective_aggregate"]["hallucination_rate"]
    if base_hall > 0:
        reduction = (base_hall - corr_hall) / base_hall * 100
        print(f"\n幻觉率：基线 {base_hall:.1%} → 纠错 {corr_hall:.1%}（降低 {reduction:.1f}%）")
    else:
        print(f"\n幻觉率：基线 {base_hall:.1%} → 纠错 {corr_hall:.1%}（基线无幻觉）")
    print(f"\n报告已保存到：{output_path}")


if __name__ == "__main__":
    main()
