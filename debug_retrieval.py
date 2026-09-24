"""Print exactly what search() retrieves for one question, so we can see
whether the chunks actually contain usable numbers or garbled table text."""
from retrieve import search

question = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
results = search(question, top_k=4, doc_filter="3M")

for i, r in enumerate(results, 1):
    print(f"\n=== Chunk {i} | {r['doc_name']} | score={r['score']:.3f} ===")
    print(r["text"])
