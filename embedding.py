import chromadb
from core import  Embedding
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, Settings
import chromadb
import argparse
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="Embedding process")
parser.add_argument("--persist-dir", type=str, default="./chroma_store",
                        help="Directory to store Chroma data")
parser.add_argument("--collection", type=str, default="emb",
                        help="Directory to store Chroma data")
parser.add_argument("--data-dir", type=str,
                    help="Directory to data folder",required=True)
parser.add_argument("--mode", type=str,
                    help="Chunking strategy", default="default")
parser.add_argument("--buffer-size",type=int,
                    help="Buffer size for semantic chunking", default=3)
parser.add_argument("--percentile",type=int,
                    help="Breakpoint percentile threshold for semantic chunking", default=95)
args = parser.parse_args()

# Setting models
Settings.embed_model = Embedding()
def default_chunking(
        persist_dir,
        collection,
        data_dir,
):
    documents = SimpleDirectoryReader(data_dir).load_data()
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter()
    nodes = []
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        for chunk in chunks:
            node = TextNode(text=chunk, metadata={})
            node.metadata["chunk_type"] = "default"
            nodes.append(node)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
def semantic_chunking(
        persist_dir,
        collection,
        data_dir,
        buffer_size=3,
        breakpoint_percentile_threshold=80

):
    documents = SimpleDirectoryReader(data_dir).load_data()
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    semantic_splitter = SemanticSplitterNodeParser(buffer_size=buffer_size, breakpoint_percentile_threshold=breakpoint_percentile_threshold)
    nodes = semantic_splitter.get_nodes_from_documents(documents)
    for node in nodes:
        node.metadata["chunk_type"] = "semantic"
    index = VectorStoreIndex(nodes, storage_context=storage_context)

## Dạng đơn giản nhất
## Có thể dùng LLM để semantic chunking, recursive chunking hoặc chunking theo khoảng cố định, hoặc chunk theo nhiều cách để có nhiều cấp độ 
if args.mode == "default":
    default_chunking(args.persist_dir,args.collection,args.data_dir)

elif args.mode == "semantic":
    semantic_chunking(args.persist_dir, args.collection,
                      args.data_dir, args.buffer_size,
                      args.percentile)
elif args.mode == "all":
    default_chunking(args.persist_dir,args.collection,args.data_dir)
    semantic_chunking(args.persist_dir, args.collection,
                      args.data_dir, args.buffer_size,
                      args.percentile)