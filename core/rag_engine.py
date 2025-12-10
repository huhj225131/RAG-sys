##RAG engine để tra cứu. Config opik để log trên web
import os
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
    
## Prompt cho lần đầu gọi llm, nếu context thu được từ db dài quá thì sẽ bị cắt ra, phần đầu
## sử dụng prompt này để hỏi LLM, phần sau dùng prompt dưới
QA_PROMPT_STR = """
Với thông tin liên quan sau đây
---------------------
{context_str}
---------------------
Với các thông tin trên, hoặc nếu không thì dựa vào chính mô hình
Câu hỏi: {query_str}
Kết quả: 
"""
default_qa_template = PromptTemplate(QA_PROMPT_STR)

##Tiếp nhận kết quả từ prompt qa hoặc prompt refine phía trước, tiếp tục thực hiện prompt
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

default_refine_template = PromptTemplate(REFINE_PROMPT_STR)
class RAGService():
    def __init__(self, node_preprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
                 similarity_top_k=3,
                 db_path="./chroma_store",
                 collection_name="emb",
                 qa_template=default_qa_template,
                 refine_template=default_refine_template):
        
        # 1. Lưu các tham số cấu hình vào self
        self.node_preprocessors = node_preprocessors
        self.similarity_top_k = similarity_top_k
        self.qa_template = qa_template
        self.refine_template = refine_template
        
        # 2. Khởi tạo DB (Chỉ làm 1 lần)
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.chroma_collection = self.db_client.get_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # 3. Load Index (Chỉ làm 1 lần trừ khi đổi Embedding Model)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=Settings.embed_model, 
        )
        
        # 4. Gọi hàm xây dựng Query Engine lần đầu
        self.rebuild_query_engine()

    def rebuild_query_engine(self):
        """Hàm này chịu trách nhiệm tạo lại Query Engine dựa trên config hiện tại"""
        
        # Tạo Synthesizer với template hiện tại
        custom_synthesizer = CustomCompactAndRefine(
            text_qa_template=self.qa_template,
            refine_template=self.refine_template
        )
        
        # Tạo lại query_engine với các tham số hiện tại (ví dụ top_k mới)
        self.query_engine = self.index.as_query_engine(
            response_synthesizer=custom_synthesizer,
            similarity_top_k=self.similarity_top_k,
            node_postprocessors=self.node_preprocessors
        )
        print(f"--> System updated with top_k={self.similarity_top_k}")

    def update_config(self, similarity_top_k=None, node_preprocessors=None, qa_template=None):
        """Hàm để user gọi từ bên ngoài khi muốn đổi config"""
        changed = False
        
        if similarity_top_k is not None:
            self.similarity_top_k = similarity_top_k
            changed = True
            
        if node_preprocessors is not None:
            self.node_preprocessors = node_preprocessors
            changed = True
            
        if qa_template is not None:
            self.qa_template = qa_template
            changed = True
            
        # Nếu có thay đổi thì build lại engine
        if changed:
            self.rebuild_query_engine()

    @track_decorator(name="RAG Query")
    def query(self, query_str):
        return self.query_engine.query(query_str)