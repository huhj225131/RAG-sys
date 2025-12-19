import os
import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from core import Embedding
from data_processing.db_manager import get_pending_documents, mark_embedded

Settings.context_window = 32000
Settings.embed_model = Embedding()
Settings.num_output = 4000

def embed_new_docs_from_db(persist_dir="./chroma_store", collection_name="emb", docstore_dir="./docstore_save"):
    rows = get_pending_documents()
    if not rows:
        return 0
    documents = []
    doc_ids = []
    
    for row in rows:
        if not row['content']: continue
        doc = Document(
            text=row['content'],
            metadata={
                "url": row['url'],
                "title": row['title'],
                "source": "web_crawl", 
                "type": "web_article",
                "crawled_at": row['created_at']
            }
        )
        documents.append(doc)
        doc_ids.append(row['id'])

    if not documents:
        return 0
    Settings.embed_model = Embedding(embed_batch_size=5)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512], 
        chunk_overlap=128
    )
    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    print(f"Tạo được {len(all_nodes)} nodes (trong đó {len(leaf_nodes)} leaf nodes).")

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
    mark_embedded(doc_ids)
    
    print(f"HOÀN TẤT!")
    return len(documents)