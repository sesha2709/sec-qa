"""Step 2: baseline evaluation on FinanceBench.

Asks Claude finance questions with filing evidence, grades each answer
with a second Claude call, and saves the run so you can track progress.

Usage:
  python eval_baseline.py --data data/financebench_open_source.jsonl --n 10
"""
import argparse
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

ANSWER_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheaper model is fine for grading


def text_of(msg) -> str:
    """Join only the text blocks, skipping thinking blocks."""
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def ask(question: str, evidence: str = "") -> str:
    content = f"Answer this finance question concisely. If you don't know, say so.\n\n"
    if evidence:
        content += f"Use this filing information to answer:\n{evidence}\n\n"
    content += f"Question: {question}"
    
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": content,
        }],
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
        max_tokens=150,
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
        # Extract evidence text from the evidence list
        evidence = ""
        if "evidence" in row and row["evidence"]:
            evidence = "\n".join(e.get("evidence_text", "") for e in row["evidence"][:1])  # Use first evidence block
        
        ans = ask(q, evidence)
        ok, judge_text = grade(q, ref, ans)
        correct += ok
        results.append({
            "id": row.get("financebench_id"),
            "question": q,
            "reference": ref,
            "answer": ans,
            "correct": ok,
            "judge": judge_text,
        })
        print(f"[{i}/{len(rows)}] {'PASS' if ok else 'FAIL'}  {q[:80]}")

    score = correct / len(rows)
    print(f"\nBaseline score: {correct}/{len(rows)} = {score:.0%}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"baseline_{int(time.time())}.json"
    out.write_text(json.dumps({"score": score, "n": len(rows), "results": results}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()