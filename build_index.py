"""Step 3a: build a searchable index from the actual 10-K PDFs.

Extracts text from PDFs in data_src/pdfs/, splits it into overlapping
chunks, embeds each chunk with Voyage AI, and saves everything to
index/chunks.json + index/embeddings.npy so retrieve.py can search it.

Usage:
  python build_index.py --pdfs data_src/pdfs --limit 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pdfplumber
import voyageai
from dotenv import load_dotenv

load_dotenv()
vo = voyageai.Client()

EMBED_MODEL = "voyage-3"
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between consecutive chunks
BATCH_SIZE = 128       # embed this many chunks per API call


def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(str(pdf_path)) as pdf:
        parts = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            parts.append(text)
            for table in page.extract_tables():
                rows = ["\t".join(str(cell or "") for cell in row) for row in table]
                parts.append("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]")
        return "\n".join(parts)


def chunk_text(text: str, doc_name: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            chunks.append({"doc_name": doc_name, "text": piece})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_batch(texts: list[str]) -> np.ndarray:
    result = vo.embed(texts, model=EMBED_MODEL, input_type="document")
    return np.array(result.embeddings, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True, help="folder containing 10-K PDFs")
    ap.add_argument("--limit", type=int, default=None, help="only process first N pdfs")
    ap.add_argument("--filter", default=None, help="only process PDFs whose name contains this substring")
    args = ap.parse_args()

    pdf_files = sorted(Path(args.pdfs).glob("*.pdf"))
    if args.filter:
        pdf_files = [p for p in pdf_files if args.filter.upper() in p.name.upper()]
    if args.limit:
        pdf_files = pdf_files[: args.limit]
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {args.pdfs}")

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Extracting {pdf_path.name} ...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text, pdf_path.stem)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding (this calls the Voyage API in batches)...")

    all_vectors = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        vecs = embed_batch([c["text"] for c in batch])
        all_vectors.append(vecs)
        print(f"  embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

    embeddings = np.vstack(all_vectors)

    Path("index").mkdir(exist_ok=True)
    Path("index/chunks.json").write_text(json.dumps(all_chunks, indent=2))
    np.save("index/embeddings.npy", embeddings)

    print(f"\nSaved {len(all_chunks)} chunks to index/chunks.json")
    print(f"Saved embeddings {embeddings.shape} to index/embeddings.npy")


if __name__ == "__main__":
    main()
