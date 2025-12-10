
from core import setup_config,RAGService,LLM_Large,LLM_Small,Embedding
import logging
from dotenv import load_dotenv
from llama_index.core import Settings
load_dotenv()

logging.getLogger("opik").setLevel(logging.ERROR)

setup_config()


# class RAGService():
#     def __init__(self, node_preprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
#                  similarity_top_k=3,
#                  db_path="./chroma_store",
#                  collection_name="emb",
#                  qa_template=default_qa_template,
#                  refine_template=default_refine_template):
Settings.llm = LLM_Large
Settings.embed_model =Embedding
LLM_Large.few_shot_custom(examples=[], system_instruction="Bạn là 1 AI giỏi :)")
rag_service = RAGService()
###
    # RAGService là lớp bao (wrapper) quản lý việc truy vấn dữ liệu từ ChromaDB và sinh câu trả lời bằng LlamaIndex.
    
    # Tính năng chính:
    # - Quản lý kết nối ChromaDB.
    # - Tích hợp logging qua Opik (tùy chọn).
    # - Hỗ trợ thay đổi cấu hình (top_k, prompt) thời gian thực mà không cần khởi động lại.
    
    # Usage:
    #     >>> from rag_service import RAGService
    #     >>> # 1. Khởi tạo
    #     >>> service = RAGService(node_preprocessors=[], similarity_top_k=3)
    #     >>> # 2. Truy vấn
    #     >>> answer = service.query("Câu hỏi là gì?")
    #     >>> # 3. Cập nhật cấu hình động
    #     >>> service.update_config(similarity_top_k=5)
###


questions = [
    "Ai là tổng thống Pháp hiện tại",
    "Xin chào nhé"
]

print("\n--- BẮT ĐẦU HỎI ---")
for q in questions:
    print(f"\nUser: {q}")
    response = rag_service.query(q)
    print(f"Bot: {response}")