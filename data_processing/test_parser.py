from llama_index.core.node_parser import MarkdownNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import os,sys
from pathlib import Path
import chromadb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Thêm vào sys.path để Python nhìn thấy các folder ở root (như utils, core...)
if project_root not in sys.path:
    sys.path.append(project_root)

from core import Embedding

load_dotenv()
Settings.embed_model=Embedding()
Settings.chunk_overlap=50
Settings.chunk_size=8000
crawl_dir = os.environ.get("DATA_CRAWL", "./crawl")
md_dir  = Path(crawl_dir) / "md"
persist_dir="./chroma_store"
collection="hackathon"
DOCSTORE_DIR = "./docstore_save"
# 0. Load dữ liệu gốc
def clean_file_metadata(file_path):
    """Chỉ lấy tên file ngắn gọn, bỏ đường dẫn dài ngoằng"""
    return {"file_name": Path(file_path).name}

documents = SimpleDirectoryReader(
    input_dir=md_dir,
    file_metadata=clean_file_metadata # <--- Mẹo 1: Chỉ lấy cái cần thiết
).load_data()

# ---------------------------------------------------------
# BƯỚC 1: Markdown Parsing & Tối ưu Metadata Header
# ---------------------------------------------------------
markdown_parser = MarkdownNodeParser()
base_nodes = markdown_parser.get_nodes_from_documents(documents)

# Mẹo 2: Duyệt qua base_nodes và gắn cờ "Cấm Embed" cho Metadata
for node in base_nodes:
    # Header Path rất tốt cho LLM hiểu ngữ cảnh, nhưng rác với Embedding
    # File Name cũng vậy
    node.excluded_embed_metadata_keys = ["file_name", "header_path"] 
    
    # Nếu bạn muốn tiết kiệm token cho LLM luôn (chỉ giữ text), dùng dòng dưới:
    # node.excluded_llm_metadata_keys = ["file_name"] 

print(f"Số lượng chương/mục lớn: {len(base_nodes)}")


# ---------------------------------------------------------
# BƯỚC 2: Cắt mịn theo phân cấp (Small-to-Big)
# Mục đích: Tạo cấu trúc Cha-Con từ các chương đã sạch sẽ ở trên
# ---------------------------------------------------------
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048,1024, 512]
)

# QUAN TRỌNG: Input ở đây phải là 'base_nodes' (kết quả bước 1)
# CHỨ KHÔNG PHẢI 'documents' nữa.
final_nodes = hierarchical_parser.get_nodes_from_documents(base_nodes)

print(f"Tổng số lượng nodes sau khi phân cấp: {len(final_nodes)}")


# ---------------------------------------------------------
# BƯỚC 3: Lọc và Lưu trữ (Indexing)
# ---------------------------------------------------------

# # Lấy các node lá (128 token) để đem đi Embed
leaf_nodes = get_leaf_nodes(final_nodes)



# # A. Lưu TẤT CẢ node (Ông, Cha, Con) vào DocStore
# # Để sau này Retriever có cái mà tra ngược lên


# # B. Chỉ Index LEAF NODE vào VectorStore
# # Để tiết kiệm tiền và search chính xác
# index = VectorStoreIndex(
#     leaf_nodes, 
#     storage_context=storage_context
# )

print("Đang Index xong theo chuẩn Pipeline: Markdown -> Hierarchical!")
db = chromadb.PersistentClient(path=persist_dir)
chroma_collection = db.get_or_create_collection(collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context.docstore.add_documents(final_nodes)
index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
storage_context.persist(persist_dir=DOCSTORE_DIR)