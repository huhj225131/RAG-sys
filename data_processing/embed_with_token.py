import os, json, argparse
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from core import Embedding

Settings.context_window = 32000
Settings.num_output = 4000

CONTENT_KEYS = ["content", "text", "body", "markdown", "md", "article", "raw", "full_text"]
TITLE_KEYS = ["title", "headline", "name"]
URL_KEYS = ["url", "link", "source_url"]


def pick_first(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw[0] == "[":
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def build_nodes(
    records: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_len: int,
    debug: bool = False,
) -> List[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: List[TextNode] = []

    kept_docs = 0
    for idx, d in enumerate(records):
        title = pick_first(d, TITLE_KEYS)
        content = pick_first(d, CONTENT_KEYS)
        url = pick_first(d, URL_KEYS)

        if debug and idx < 3:
            print(f"\n[DEBUG sample #{idx}] keys={list(d.keys())[:20]}")
            print(f"title={title[:80]}")
            print(f"url={url[:80]}")
            print(f"content_len={len(content)}")

        if len(content) < min_len:
            continue

        kept_docs += 1
        full_text = f"{title}\n\n{content}" if title else content

        chunks = splitter.split_text(full_text)
        for j, chunk in enumerate(chunks):
            meta = {
                "source": d.get("source", "baochinhphu"),
                "domain": d.get("domain", "admin"),
                "url": url,
                "title": title,
                "chunk_type": "default",
                "doc_index": idx,
                "chunk_index": j,
            }
            nodes.append(TextNode(text=chunk, metadata=meta))

    print(f"Loaded records={len(records)}, kept_docs(min_len={min_len})={kept_docs}, nodes={len(nodes)}")
    return nodes


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--persist-dir", type=str, default="./chroma_store")
    parser.add_argument("--collection", type=str, default="admin_emb")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-len", type=int, default=50, help="lọc doc")
    parser.add_argument("--debug", action="store_true", help="sample")
    args = parser.parse_args()

    Settings.embed_model = Embedding(embed_batch_size=5)

    records = load_jsonl_or_json(args.jsonl)
    if not records:
        raise RuntimeError("Không parse được.")

    nodes = build_nodes(
        records=records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_len=args.min_len,
        debug=args.debug,
    )
    if not nodes:
        raise RuntimeError("Không tạo được node")

    db = chromadb.PersistentClient(path=args.persist_dir)
    chroma_collection = db.get_or_create_collection(args.collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"Done. collection='{args.collection}' persist_dir='{args.persist_dir}'")


if __name__ == "__main__":
    main()
