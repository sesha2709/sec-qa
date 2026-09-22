# sec-qa

RAG over SEC filings with a real eval suite. Baseline first, then improve.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then paste your key into .env
python hello.py
```

## Baseline eval
```bash
git clone https://github.com/patronus-ai/financebench.git data_src
mkdir -p data && cp data_src/data/financebench_open_source.jsonl data/
python eval_baseline.py --data data/financebench_open_source.jsonl --n 10
```

## Log
| Date | Change | Score |
|------|--------|-------|
|      | Closed-book baseline | |
