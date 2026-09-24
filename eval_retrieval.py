"""Step 3c: real RAG evaluation on FinanceBench.

Instead of using the dataset's pre-picked evidence (the "oracle" setup),
this retrieves evidence itself from the indexed PDFs using retrieve.py,
then answers and grades exactly like eval_baseline.py.

Requires index/chunks.json + index/embeddings.npy to exist already
(run build_index.py first).

Usage:
  python eval_retrieval.py --data data/financebench_open_source.jsonl --n 10
"""
import argparse
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from retrieve import search

load_dotenv()
client = anthropic.Anthropic()

ANSWER_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 8


def text_of(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def ask(question: str, evidence: str) -> str:
    content = (
        "Answer this finance question concisely using the filing excerpts below. "
        "If the excerpts don't contain the answer, say so.\n\n"
        f"Filing excerpts:\n{evidence}\n\n"
        f"Question: {question}"
    )
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
    )
    return text_of(msg)


def grade(question: str, reference: str, answer: str) -> tuple[bool, str]:
    prompt = f"""You are grading an answer against a reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Is the candidate answer consistent with the reference (same key facts and numbers)?
Reply on the first line with exactly CORRECT or INCORRECT, then one short sentence why."""
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = text_of(msg)
    first = text.splitlines()[0].strip().upper() if text else ""
    return first.startswith("CORRECT"), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[: args.n]

    results, correct = [], 0
    for i, row in enumerate(rows, 1):
        q, ref = row["question"], row["answer"]

        # Scope retrieval to the filing's company, using the dataset's own
        # doc_name metadata (e.g. "3M_2018_10K" -> "3M"). This mirrors how
        # real RAG systems narrow the search space before doing semantic
        # search, instead of searching the entire multi-company corpus.
        company = row.get("doc_name", "").split("_")[0]
        retrieved = search(q, top_k=TOP_K, doc_filter=company)
        evidence = "\n\n---\n\n".join(
            f"(from {r['doc_name']})\n{r['text']}" for r in retrieved
        )

        ans = ask(q, evidence)
        ok, judge_text = grade(q, ref, ans)
        correct += ok
        results.append({
            "id": row.get("financebench_id"),
            "question": q,
            "reference": ref,
            "answer": ans,
            "retrieved_docs": [r["doc_name"] for r in retrieved],
            "retrieval_scores": [round(r["score"], 3) for r in retrieved],
            "correct": ok,
            "judge": judge_text,
        })
        print(f"[{i}/{len(rows)}] {'PASS' if ok else 'FAIL'}  {q[:80]}")

    score = correct / len(rows)
    print(f"\nRetrieval score: {correct}/{len(rows)} = {score:.0%}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"retrieval_{int(time.time())}.json"
    out.write_text(json.dumps({"score": score, "n": len(rows), "top_k": TOP_K, "results": results}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
