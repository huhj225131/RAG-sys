from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import os, sys
from pathlib import Path
import chromadb

# --- SETUP CƠ BẢN ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core import Embedding

load_dotenv()
Settings.embed_model = Embedding()
# Lưu ý: Settings.chunk_size ở đây chỉ là default, Hierarchical sẽ dùng tham số riêng bên dưới

crawl_dir = os.environ.get("DATA_CRAWL", "./crawl")
md_dir  = Path(crawl_dir) / "md"
persist_dir = "./chroma_store"
collection_name = "hackathon"
DOCSTORE_DIR = "./docstore_save"
DONE_FILE = md_dir / "done.txt"

# =========================================================
# BƯỚC 0: LỌC FILE ĐÃ LÀM (GIỮ NGUYÊN)
# =========================================================
processed_files = set()
if DONE_FILE.exists():
    with open(DONE_FILE, "r", encoding="utf-8") as f:
        processed_files = {line.strip() for line in f if line.strip()}

print(f"--> Đã tìm thấy {len(processed_files)} file đã xử lý trước đó.")

all_md_files = list(md_dir.glob("*.md"))
new_files_to_process = [f for f in all_md_files if f.name not in processed_files]

if not new_files_to_process:
    print("✅ Không có file mới. Hệ thống nghỉ!")
    sys.exit(0)

print(f"🚀 Tìm thấy {len(new_files_to_process)} file mới. Bắt đầu xử lý...")

# =========================================================
# LOAD DATA
# =========================================================
def clean_file_metadata(file_path):
    return {"file_name": Path(file_path).name}

documents = SimpleDirectoryReader(
    input_files=new_files_to_process, 
    file_metadata=clean_file_metadata
).load_data()

# =========================================================
# BƯỚC 1 & 2 GỘP LẠI: CẮT TRỰC TIẾP (BỎ MARKDOWN PARSER)
# =========================================================
print("--> Đang cấu hình Hierarchical Node Parser...")

hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 512], # Cha 1024, Con 512
    
    # --- QUAN TRỌNG NHẤT: CHỐNG MẤT DỮ LIỆU ---
    # 128 token overlap (~50-70 từ).
    # Đảm bảo đoạn cuối node trước và đoạn đầu node sau giống hệt nhau.
    # Không bao giờ sợ bị cắt giữa chừng làm mất nghĩa.
    chunk_overlap=128 
)

print("--> Đang cắt nodes từ documents gốc...")
# Input trực tiếp là 'documents' (chứa toàn bộ nội dung file)
final_nodes = hierarchical_parser.get_nodes_from_documents(documents)

print(f"✅ Tổng số lượng nodes (Cha + Con) sau khi cắt: {len(final_nodes)}")

# =========================================================
# BƯỚC 3: LƯU TRỮ (GIỮ NGUYÊN)
# =========================================================
leaf_nodes = get_leaf_nodes(final_nodes)

db = chromadb.PersistentClient(path=persist_dir)
chroma_collection = db.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if os.path.exists(DOCSTORE_DIR) and os.path.exists(os.path.join(DOCSTORE_DIR, "docstore.json")):
    print("--> Load DocStore cũ...")
    storage_context = StorageContext.from_defaults(
        persist_dir=DOCSTORE_DIR, 
        vector_store=vector_store
    )
else:
    print("--> Tạo DocStore mới...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

storage_context.docstore.add_documents(final_nodes)

print("--> Đang embedding và lưu vào Chroma...")
index = VectorStoreIndex(
    leaf_nodes, 
    storage_context=storage_context,
    show_progress=True 
)

storage_context.persist(persist_dir=DOCSTORE_DIR)
print("✅ Đã lưu dữ liệu thành công!")

# =========================================================
# CẬP NHẬT DONE.TXT
# =========================================================
with open(DONE_FILE, "a", encoding="utf-8") as f:
    for file_path in new_files_to_process:
        f.write(f"{file_path.name}\n")

print("Hoàn tất!")