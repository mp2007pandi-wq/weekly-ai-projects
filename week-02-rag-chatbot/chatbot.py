"""RAG chatbot: BM25 retrieval + chat memory + citations."""
import math
import os
import re
from collections import Counter
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"
STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "it", "its", "what", "about", "how", "do", "does", "for", "on", "with", "that", "this"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def chunk(text, size=50, overlap=10):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


def load_chunks():
    out = []
    for f in sorted(DOCS_DIR.glob("*.txt")):
        for n, c in enumerate(chunk(f.read_text(encoding="utf-8")), 1):
            out.append({"source": f"{f.name}#{n}", "text": c})
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k=2):
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def contextualize(question, history):
    """Short follow-ups borrow terms from the previous user question."""
    if history and len(tokenize(question)) <= 3:
        return history[-1]["user"] + " " + question
    return question


def answer(question, hits, history):
    ctx = "\n".join(f"[{c['source']}] {c['text']}" for _, c in hits)
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        msgs = [{"role": "system", "content": "Answer only from the context and cite sources like [file#n]."}]
        for h in history[-3:]:
            msgs += [{"role": "user", "content": h["user"]}, {"role": "assistant", "content": h["bot"]}]
        msgs.append({"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"})
        r = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=msgs)
        return r.choices[0].message.content
    return "(offline mode, best passage)\n" + hits[0][1]["text"]


def main():
    index = BM25(load_chunks())
    history = []
    print("RAG chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        hits = index.search(contextualize(q, history))
        if not hits:
            reply = "I couldn't find anything relevant in the documents."
        else:
            reply = answer(q, hits, history)
            reply += "\n\nSources: " + ", ".join(c["source"] for _, c in hits)
        print("\nBot:", reply)
        history.append({"user": q, "bot": reply})


if __name__ == "__main__":
    main()
