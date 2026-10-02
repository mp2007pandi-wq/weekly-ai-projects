# Week 1: Mini RAG

A minimal Retrieval-Augmented Generation pipeline with no heavy dependencies.

## How it works
1. **Load** .txt files from `docs/`.
2. **Chunk** them into overlapping word windows.
3. **Retrieve** top-k chunks with TF-IDF + cosine similarity (pure Python).
4. **Generate**: builds a grounded prompt. If `OPENAI_API_KEY` is set it calls an LLM; otherwise it prints the retrieved context.

## Run
```bash
cd week-01-mini-rag
python rag.py "What is RAG?"
```

Optional LLM: `pip install openai` and set `OPENAI_API_KEY` in your environment.

## Next ideas
Embeddings + vector DB (FAISS/Chroma), PDF loading, reranking.
