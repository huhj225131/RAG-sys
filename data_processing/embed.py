import os, json, argparse
from pathlib import Path
from typing import Dict, Any, List
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
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
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            pass 
    
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

def create_documents(records: List[Dict[str, Any]], min_len: int, dataset_tag: str) -> List[Document]:
    docs = []
    kept_count = 0
    
    for idx, d in enumerate(records):
        title = pick_first(d, TITLE_KEYS)
        content = pick_first(d, CONTENT_KEYS)
        url = pick_first(d, URL_KEYS)

        if len(content) < min_len:
            continue

        kept_count += 1
        full_text = f"{title}\n\n{content}" if title else content

        meta = {
            "source": d.get("source", dataset_tag),
            "domain": d.get("domain", "admin"),
            "url": url,
            "title": title,
            "dataset": dataset_tag,  
            "doc_origin_index": idx
        }
        
        doc = Document(text=full_text, metadata=meta)
        docs.append(doc)
    return docs

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--persist-dir", type=str, default="./chroma_store")
    parser.add_argument("--collection", type=str, default="emb")
    parser.add_argument("--docstore-dir", type=str, default="./docstore_save")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--tag", type=str, default="general", help="")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[1024, 512], help="")
    parser.add_argument("--chunk-overlap", type=int, default=128, help="")
    parser.add_argument("--min-len", type=int, default=50)
    args = parser.parse_args()

    Settings.embed_model = Embedding(embed_batch_size=5) 

    records = load_jsonl_or_json(args.jsonl)
    if not records:
        raise RuntimeError("File dữ liệu rỗng hoặc sai format.")
    documents = create_documents(records, args.min_len, dataset_tag=args.tag)
    if not documents:
        raise RuntimeError("Không có document nào hợp lệ.")

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=args.chunk_sizes,
        chunk_overlap=args.chunk_overlap
    )

    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    db = chromadb.PersistentClient(path=args.persist_dir)
    chroma_collection = db.get_or_create_collection(args.collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    docstore_path = os.path.join(args.docstore_dir, "docstore.json")
    
    if os.path.exists(docstore_path):
        storage_context = StorageContext.from_defaults(
            persist_dir=args.docstore_dir, 
            vector_store=vector_store
        )
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(all_nodes)
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        show_progress=True
    )
    storage_context.persist(persist_dir=args.docstore_dir)
    
    print(f"HOÀN TẤT!")

if __name__ == "__main__":
    main()
# python -m data_processing.embed --jsonl data_processing/data_raw/admin/docs.jsonl --persist-dir ./chroma_store --docstore-dir ./docstore_save --collection emb --tag "dia_ly_hanh_chinh"