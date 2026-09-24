"""Step 3b: retrieval helpers, used by eval_retrieval.py.

Loads the index built by build_index.py, embeds a question, and returns
the top-k most similar chunks by cosine similarity.
"""
import json
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()
vo = voyageai.Client()

EMBED_MODEL = "voyage-3"

_chunks = None
_embeddings = None


def _load():
    global _chunks, _embeddings
    if _chunks is None:
        _chunks = json.loads(Path("index/chunks.json").read_text())
        _embeddings = np.load("index/embeddings.npy")
        # normalize once so we can use a plain dot product as cosine similarity
        _embeddings = _embeddings / np.linalg.norm(_embeddings, axis=1, keepdims=True)
    return _chunks, _embeddings


def embed_query(question: str) -> np.ndarray:
    result = vo.embed([question], model=EMBED_MODEL, input_type="query")
    vec = np.array(result.embeddings[0], dtype=np.float32)
    return vec / np.linalg.norm(vec)


def search(question: str, top_k: int = 4, doc_filter: str = None) -> list[dict]:
    chunks, embeddings = _load()
    q_vec = embed_query(question)
    scores = embeddings @ q_vec  # cosine similarity, since both sides are normalized

    if doc_filter:
        mask = np.array([doc_filter.upper() in c["doc_name"].upper() for c in chunks])
        scores = np.where(mask, scores, -1.0)  # exclude non-matching docs entirely

    top_idx = np.argsort(-scores)[:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in top_idx
    ]


if __name__ == "__main__":
    # quick manual test: python retrieve.py "your question here"
    import sys
    q = " ".join(sys.argv[1:]) or "What was 3M's capital expenditure?"
    for r in search(q):
        print(f"[{r['score']:.3f}] {r['doc_name']}: {r['text'][:150]}...")
