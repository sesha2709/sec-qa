"""Check which doc_names exist in the index vs what the eval needs."""
import json
from pathlib import Path
from collections import Counter

chunks = json.loads(Path("index/chunks.json").read_text())
counts = Counter(c["doc_name"] for c in chunks)

targets = ["3M_2018", "3M_2022", "ACTIVISION", "3M_2019"]
print("Matching doc_names in index:")
for t in targets:
    matches = {k: v for k, v in counts.items() if t.upper() in k.upper()}
    print(f"  {t}: {matches if matches else 'NOT FOUND'}")
