
from core import setup_config
from core import RAGService
import logging
from dotenv import load_dotenv
load_dotenv()

logging.getLogger("opik").setLevel(logging.ERROR)

setup_config(small=False)


rag_service = RAGService()


questions = [
    "Ai là tổng thống Pháp hiện tại",
    "Xin chào nhé"
]

print("\n--- BẮT ĐẦU HỎI ---")
for q in questions:
    print(f"\nUser: {q}")
    response = rag_service.query(q)
    print(f"Bot: {response}")