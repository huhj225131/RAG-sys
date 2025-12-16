##RAG engine để tra cứu. Config opik để log trên web
import os
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex,PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings, ChatPromptTemplate
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
import chromadb,re
from .custom_synthesizer import CustomCompactAndRefine
from abc import ABC, abstractmethod



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
Dưới đây là các thông tin ngữ cảnh được cung cấp:
---------------------
{context_str}
---------------------

Dựa vào ngữ cảnh trên, hãy trả lời câu hỏi trắc nghiệm sau.
Quy tắc (không nhắc lại các quy tắc):
1. Tìm kiếm thông tin trong ngữ cảnh (nếu có) để trả lời
2. Nếu không có ngữ cảnh thì tự trả lời
3. Đưa ra giải thích ngắn gọn vì sao chọn đáp án đó
4. Với câu hỏi có nội dung bạo lực, nhạy cảm, bắt buộc chọn đáp án có nội dung không trả lời câu hỏi này
5. BẮT BUỘC phải kết thúc bằng dòng chính xác: "Đáp án: <Ký tự>" Ký tự là 1 chứ cái tiếng Anh đại diện cho đáp án
6. Nếu không có câu trả lời, bắt buộc trả lời: "Đáp án: 1"
Câu hỏi: {query_str}
Giải thích
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

Dựa vào ngữ cảnh trên, hãy trả lời câu hỏi trắc nghiệm sau.
Quy tắc (không nhắc lại các quy tắc):
1. Tìm kiếm thông tin trong ngữ cảnh (nếu có) để trả lời
2. Nếu không có ngữ cảnh thì tự trả lời
3. Đưa ra giải thích ngắn gọn vì sao chọn đáp án đó
4. Với câu hỏi có nội dung bạo lực, nhạy cảm, bắt buộc chọn đáp án có nội dung không trả lời câu hỏi này
5. BẮT BUỘC phải kết thúc bằng dòng chính xác: "Đáp án: <Ký tự>" Ký tự là 1 chứ cái tiếng Anh đại diện cho đáp án
6. Nếu không có câu trả lời, bắt buộc trả lời: "Đáp án: 1"

Giải thích:
"""

default_refine_template = PromptTemplate(REFINE_PROMPT_STR)

SIDEKICK_PROMPT_STR = """
Hãy trả lời câu hỏi trắc nghiệm sau.
Quy tắc (không nhắc lại các quy tắc):
1. Đưa ra giải thích ngắn gọn vì sao chọn đáp án đó
2. Nếu là câu hỏi toán học, hãy giải thích từng bước, không cần phải ngắn gọn
3. Với câu hỏi có nội dung bạo lực, nhạy cảm, bắt buộc chọn đáp án có nội dung không trả lời câu hỏi này
4. BẮT BUỘC phải kết thúc bằng dòng chính xác: "Đáp án: <Ký tự>" Ký tự là 1 chứ cái tiếng Anh đại diện cho đáp án
5. Nếu không chắc chắn biết rõ đáp án,TUYỆT ĐỐI BẮT BUỘC phải đưa ra câu trả lời: "Đáp án: 1"
Câu hỏi: {query_str}
Giải thích
"""
sidekick_template = PromptTemplate(SIDEKICK_PROMPT_STR)
class RAGService(ABC):
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
    @abstractmethod
    def rebuild_query_engine(self):
        """Hàm này chịu trách nhiệm tạo lại Query Engine dựa trên config hiện tại"""
        
        pass

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
    
class SimpleRAGService(RAGService):
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



def extract_answer(text: str, valid_letters=None) -> str:
    m = re.search(r"Đáp án:\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if not m:
        return False

    ans = m.group(1).upper()
    if valid_letters is not None and ans not in valid_letters:
        return False
    return True
class V2RAGService(RAGService):
    def __init__(self, docstore_dir, 
                 node_preprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
                 similarity_top_k=3,
                 db_path="./chroma_store",
                 collection_name="hackathon",
                 qa_template=default_qa_template,
                 refine_template=default_refine_template,
                 sidekick=sidekick_template):
        
        self.sidekick = sidekick
        self.docstore_dir = docstore_dir
        self.storage_context = None 
        
        
        super().__init__(node_preprocessors=node_preprocessors,
                         similarity_top_k=similarity_top_k,
                         db_path=db_path,
                         collection_name=collection_name,
                         qa_template=qa_template,
                         refine_template=refine_template)

    def rebuild_query_engine(self):
        """
        Override lại hàm này để dùng AutoMergingRetriever thay vì Retriever thường
        """
        
        # 1. Load StorageContext (Chỉ load 1 lần nếu chưa có)
        # AutoMergingRetriever bắt buộc cần cái này để map từ Node Con -> Node Cha
        if self.storage_context is None:
            print(f"Loading Storage Context from {self.docstore_dir}...")
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.docstore_dir,
                vector_store=self.vector_store # Tái sử dụng vector store từ lớp cha
            )

        # 2. Tạo Base Retriever (Tìm kiếm Node Con - Leaf Nodes)
        base_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        # 3. Tạo AutoMergingRetriever (Wrapper quản lý việc gộp Cha-Con)
        self.retriever = AutoMergingRetriever(
            base_retriever, 
            storage_context=self.storage_context, 
            verbose=True # Bật log để xem process merge
        )

        # 4. Tạo Synthesizer (Bộ tổng hợp câu trả lời)
        custom_synthesizer = CustomCompactAndRefine(
            text_qa_template=self.qa_template,
            refine_template=self.refine_template
        )

        # 5. Tạo Query Engine thủ công
        # Vì cấu trúc Retriever phức tạp, ta dùng RetrieverQueryEngine thay vì as_query_engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            response_synthesizer=custom_synthesizer,
            node_postprocessors=self.node_preprocessors
        )
    @track_decorator(name="RAG Query")

    def query(self, query_str):

        first_ouput = Settings.llm.complete(self.sidekick.format(query_str= query_str))
        if extract_answer(first_ouput):
            return first_ouput
        return self.query_engine.query(query_str)

