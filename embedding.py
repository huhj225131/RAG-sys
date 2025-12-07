import chromadb
from model import  Embedding
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
import chromadb
import argparse

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="Embedding process")
parser.add_argument("--persist-dir", type=str, default="./chroma_store",
                        help="Directory to store Chroma data")
parser.add_argument("--collection", type=str, default="emb",
                        help="Directory to store Chroma data")
parser.add_argument("--data-dir", type=str,
                    help="Directory to data folder",required=True)
args = parser.parse_args()

# Setting models
Settings.embed_model = Embedding()

documents = SimpleDirectoryReader(args.data_dir).load_data()
db = chromadb.PersistentClient(path=args.persist_dir)
chroma_collection = db.get_or_create_collection(args.collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
## Dạng đơn giản nhất
## Có thể dùng LLM để semantic chunking, recursive chunking hoặc chunking theo khoảng cố định, hoặc chunk theo nhiều cách để có nhiều cấp độ 