# Week 2: RAG Chatbot with Memory and Citations

An interactive chatbot that answers from your local documents and cites its sources. Pure Python, no dependencies.

## Features
- **BM25 retrieval**, a stronger keyword ranker than plain TF-IDF.
- **Chat memory**: follow-up questions ("what about its cost?") reuse terms from the previous question.
- **Citations**: every answer lists the source file and chunk.
- **Optional LLM**: set `OPENAI_API_KEY` (and `pip install openai`) for generated answers. Otherwise it shows the best passages.

## Run
```bash
cd week-02-rag-chatbot
python chatbot.py
```
Type `exit` to quit. Put your own `.txt` files in `docs/`.

## Next ideas
Embeddings + vector DB, PDF loading, reranking, a web UI.
