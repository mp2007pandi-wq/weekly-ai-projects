"""Minimal RAG: chunk -> TF-IDF retrieve -> grounded prompt -> (optional) LLM."""
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk(text, size=60, overlap=15):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


def load_chunks():
    chunks = []
    for f in sorted(DOCS_DIR.glob("*.txt")):
        for c in chunk(f.read_text(encoding="utf-8")):
            chunks.append((f.name, c))
    return chunks


class TfidfIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tfs = [Counter(tokenize(c)) for _, c in chunks]
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(tf) for tf in self.tfs]

    def _vec(self, tf):
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=2):
        q = self._vec(Counter(tokenize(query)))
        scored = [(sum(w * v.get(t, 0.0) for t, w in q.items()), i) for i, v in enumerate(self.vecs)]
        scored.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scored[:k] if s > 0]


def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{src}] {text}" for _, (src, text) in hits)
    return f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def main():
    question = " ".join(sys.argv[1:]) or "What is RAG?"
    hits = TfidfIndex(load_chunks()).search(question)
    if not hits:
        print("No relevant context found.")
        return
    prompt = build_prompt(question, hits)
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        print(r.choices[0].message.content)
    else:
        print("(No OPENAI_API_KEY set; showing retrieved context)\n")
        print(prompt)


if __name__ == "__main__":
    main()
