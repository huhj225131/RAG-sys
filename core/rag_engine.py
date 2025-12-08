import opik,os
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex,PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings, ChatPromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
import chromadb
from .custom_synthesizer import CustomCompactAndRefine



enable_opik = os.getenv("ENABLE_OPIK", "False").lower() == "true"

if enable_opik:
    import opik
    # Nếu BẬT: Dùng decorator thật của Opik
    track_decorator = opik.track
else:
    # Nếu TẮT: Tạo một decorator giả (không làm gì cả)
    def track_decorator(name=None, **kwargs):
        def decorator(func):
            return func 
        return decorator
QA_PROMPT_STR = """
Với thông tin liên quan sau đây
---------------------
{context_str}
---------------------
Với các thông tin trên, hoặc nếu không thì dựa vào chính mô hình
Câu hỏi: {query_str}
Kết quả: 
"""
qa_template = PromptTemplate(QA_PROMPT_STR)
REFINE_PROMPT_STR = """
Câu hỏi gốc là: {query_str}
Chúng tôi đã có một câu trả lời dự kiến: {existing_answer}

Chúng tôi có thêm thông tin ngữ cảnh dưới đây:
---------------------
{context_msg}
---------------------

Dựa trên ngữ cảnh mới này và kiến thức hiện có, hãy cập nhật hoặc tinh chỉnh câu trả lời dự kiến để nó đầy đủ và chính xác hơn.
Nếu ngữ cảnh mới không hữu ích hoặc không liên quan, hãy giữ nguyên câu trả lời dự kiến.

Kết quả đã tinh chỉnh: 
"""

refine_template = PromptTemplate(REFINE_PROMPT_STR)
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
        custom_synthesizer = CustomCompactAndRefine(
            text_qa_template=qa_template,
            refine_template=refine_template
        )
        self.query_engine = self.index.as_query_engine(response_synthesizer=custom_synthesizer,
                                                       similarity_top_k=3,
                                                       node_postprocessors=[
                                            SimilarityPostprocessor(similarity_cutoff=0.7)])

    @track_decorator(name="RAG Query")
    def query(self, query_str):
        # Hàm này dùng lại self.query_engine đã load ở trên
        query_result = self.query_engine.query(query_str)
        return query_result