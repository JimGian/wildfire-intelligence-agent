"""
Build the RAG index from docs/*.md files.

Run: python wildfire_agent/rag/build_index.py

Outputs:
  wildfire_agent/rag/index.npz   — float32 matrix (N x 384)
  wildfire_agent/rag/docs.json   — list of {title, location, year, type, source, text, filename}

Idempotent: always rebuilds from scratch.
"""

import json
import re
from pathlib import Path

import numpy as np

DOCS_DIR  = Path(__file__).parent / "docs"
INDEX_OUT = Path(__file__).parent / "index.npz"
DOCS_OUT  = Path(__file__).parent / "docs.json"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_doc(path: Path) -> dict | None:
    raw = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(raw)
    if not m:
        print(f"  SKIP {path.name} — no YAML frontmatter")
        return None

    fm_block = m.group(1)
    body = raw[m.end():].strip()
    if not body:
        print(f"  SKIP {path.name} — empty body")
        return None

    meta: dict = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')

    for required in ("title", "source"):
        if required not in meta:
            print(f"  SKIP {path.name} — missing required field '{required}'")
            return None

    return {
        "title":    meta.get("title", ""),
        "location": meta.get("location", ""),
        "year":     meta.get("year", ""),
        "type":     meta.get("type", ""),
        "source":   meta.get("source", ""),
        "text":     body,
        "filename": path.name,
    }


def build():
    from sentence_transformers import SentenceTransformer

    docs = []
    for md_path in sorted(DOCS_DIR.glob("*.md")):
        doc = _parse_doc(md_path)
        if doc:
            docs.append(doc)
            print(f"  OK  {md_path.name} — {len(doc['text'])} chars")

    if not docs:
        raise RuntimeError("No valid docs found in docs/")

    print(f"\nLoading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [f"{d['title']} {d['location']} {d['text']}" for d in docs]
    print(f"Encoding {len(texts)} documents ...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    np.savez_compressed(INDEX_OUT, embeddings=embeddings)
    DOCS_OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {INDEX_OUT}  shape={embeddings.shape}")
    print(f"Saved: {DOCS_OUT}  ({len(docs)} docs)")
    return docs, embeddings


if __name__ == "__main__":
    build()
