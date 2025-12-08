import opik,os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
import chromadb

enable_opik = os.getenv("ENABLE_OPIK", "False").lower() == "true"

if enable_opik:
    import opik
    # Nếu BẬT: Dùng decorator thật của Opik
    track_decorator = opik.track
else:
    # Nếu TẮT: Tạo một decorator giả (không làm gì cả)
    def track_decorator(name=None, **kwargs):
        def decorator(func):
            return func # Trả về nguyên hàm gốc, không track gì hết
        return decorator
class RAGService:
    def __init__(self, db_path="./chroma_store", collection_name="emb"):
        
        # Kết nối DB
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.chroma_collection = self.db_client.get_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Load Index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=Settings.embed_model, 
        )
        
        # Tạo Query Engine sẵn luôn
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    @track_decorator(name="RAG Query Loop")
    def query(self, query_str):
        # Hàm này dùng lại self.query_engine đã load ở trên
        query_result = self.query_engine.query(query_str)
    
        # Kiểm tra nếu trả về rỗng
        if not query_result.response or query_result.response == "Empty Response":
            messages = [ChatMessage(role="user", content=query_str)]
            return Settings.llm.chat(messages)
            
        return query_result