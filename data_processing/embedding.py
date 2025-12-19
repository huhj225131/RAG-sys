import json
import chromadb
from core import Embedding
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from data_processing.db_manager import get_pending_documents, mark_embedded

Settings.context_window = 32000
Settings.embed_model = Embedding()
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

def _get_storage_context(persist_dir: str, collection_name: str) -> StorageContext:
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return StorageContext.from_defaults(vector_store=vector_store)

def embed_new_docs_from_db(persist_dir="./chroma_store", collection_name="emb"):
    rows = get_pending_documents()
    if not rows:
        print("[Embedding] Không có document mới (pending=0).")
        return 0

    documents = []
    doc_ids = []

    for row in rows:
        doc = Document(
            text=row["content"] or "",
            metadata={
                "url": row["url"] or "",
                "title": row["title"] or "",
                "source": "web_crawl",
                "crawled_at": row["created_at"],
                "doc_id": row["id"], 
            },
        )
        documents.append(doc)
        doc_ids.append(row["id"])

    if not documents:
        print("[Embedding] Pending rows rỗng sau khi convert")
        return 0

    storage_context = _get_storage_context(persist_dir, collection_name)

    splitter = SentenceSplitter()
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    mark_embedded(doc_ids)

    print(f"[Embedding] Đã embed xong {len(documents)} tài liệu mới và mark_embedded.")
    return len(documents)

def embed_jsonl_file(jsonl_path: str, persist_dir="./chroma_store", collection_name="emb"):
    print(f"[Embedding] Đang xử lý file: {jsonl_path}")

    documents = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                doc = Document(
                    text=data.get("text", ""),
                    metadata={
                        "url": data.get("url", ""),
                        "title": data.get("title", ""),
                        "source": data.get("source", ""),
                        "crawled_at": data.get("crawled_at", ""),
                    },
                )
                documents.append(doc)
    except FileNotFoundError:
        print("File JSONL không tồn tại.")
        return 0

    if not documents:
        print("Không có document nào để embed.")
        return 0

    storage_context = _get_storage_context(persist_dir, collection_name)

    splitter = SentenceSplitter()
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"[Embedding] Đã embed xong {len(documents)} tài liệu vào collection '{collection_name}'")
    return len(documents)
