import os
import argparse
from pathlib import Path
from typing import List
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from core import Embedding

Settings.context_window = 32000
Settings.num_output = 4000

def _create_and_persist_index(documents: List[Document], persist_dir: str, collection_name: str, docstore_dir: str):
    if not documents:
        print("Không có tài liệu nào để xử lý.")
        return

    Settings.embed_model = Embedding(embed_batch_size=5) 

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512], 
        chunk_overlap=128
    )
    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if not os.path.exists(docstore_dir):
        os.makedirs(docstore_dir, exist_ok=True)
    
    docstore_path = os.path.join(docstore_dir, "docstore.json")
    
    if os.path.exists(docstore_path):
        storage_context = StorageContext.from_defaults(
            persist_dir=docstore_dir, 
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

    storage_context.persist(persist_dir=docstore_dir)
    print(f"Đã lưu Vector vào '{persist_dir}' và Docstore vào '{docstore_dir}'")

def process_md_list(file_paths: list, persist_dir="./chroma_store", collection="emb", docstore_dir="./docstore_save"):
    documents = []
    for f_path in file_paths:
        p = Path(f_path)
        try:
            content = p.read_text(encoding="utf-8").strip()
            if content:
                doc = Document(
                    text=content,
                    metadata={
                        "source": p.name, 
                        "title": p.stem, 
                        "type": "upload_md",
                        "domain": "markdown_import"
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Lỗi đọc file MD {p.name}: {e}")
    
    if not documents: 
        return 0

    _create_and_persist_index(documents, persist_dir, collection, docstore_dir)
    return len(documents)

def load_md_files_from_folder(folder_path: str, dataset_tag: str) -> List[Document]:
    p = Path(folder_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder_path}")

    md_files = list(p.rglob("*.md"))
    docs = []

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content: continue
            
            title = file_path.stem
            meta = {
                "source": file_path.name,
                "domain": "markdown_import",
                "url": file_path.name,
                "title": title,
                "dataset": dataset_tag,
            }

            full_text = f"# {title}\n\n{content}"
            docs.append(Document(text=full_text, metadata=meta))
            
        except Exception as e:
            print(f"Lỗi đọc file {file_path.name}: {e}")
    return docs

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--persist-dir", type=str, default="./chroma_store")
    parser.add_argument("--collection", type=str, default="emb") 
    parser.add_argument("--docstore-dir", type=str, default="./docstore_save")
    parser.add_argument("--folder", type=str, required=True, help="")
    parser.add_argument("--tag", type=str, default="tai_lieu_md", help="")

    args = parser.parse_args()
    documents = load_md_files_from_folder(args.folder, dataset_tag=args.tag)
    if not documents:
        raise RuntimeError("Không có document nào để xử lý.")
        
    _create_and_persist_index(documents, args.persist_dir, args.collection, args.docstore_dir)
    print("HOÀN TẤT!")

if __name__ == "__main__":
    main()