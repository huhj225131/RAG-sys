
from core import SimpleRAGService,LLM_Large,LLM_Small,Embedding
from metric import setup_opik
import logging
from dotenv import load_dotenv
import os,json,random,csv
from llama_index.core import Settings
load_dotenv()

logging.getLogger("opik").setLevel(logging.ERROR)

setup_opik()

Settings.context_window=32000
Settings.num_output=4000
Settings.chunk_size=2048
Settings.chunk_overlap=200
# class RAGService():
#     def __init__(self, node_preprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
#                  similarity_top_k=3,
#                  db_path="./chroma_store",
#                  collection_name="emb",
#                  qa_template=default_qa_template,
#                  refine_template=default_refine_template):
Settings.llm = LLM_Large()
Settings.embed_model =Embedding()
rag_service = SimpleRAGService()
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
filename="few_shot.json"
base_dir = os.getenv("DATA_DIR", "./data")
full_path = os.path.join(base_dir, filename)
if not os.path.exists(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")
with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
few_shot = []
for i in range(2):
        index = random.randint(0, len(data) -1)
        few_shot.append((f"{data[index]["question"]}\nLựa chọn: {data[index]["choices"]}",f"Giải thích: {data[index]["explanation"]}\nĐáp án: {data[index]["answer"]}"))
Settings.llm.few_shot_custom(examples=few_shot, system_instruction="Bạn đang giải câu hỏi trắc nghiệm")


filename="test.json"
base_dir = os.getenv("DATA_DIR", "./data")
full_path = os.path.join(base_dir, filename)
if not os.path.exists(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")
with open(full_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
print("\n--- BẮT ĐẦU HỎI ---")



import csv
import os

CHECKPOINT_FILE = "checkpoint.txt"
RESULT_FILE = "result.csv"

# --- HÀM ĐỌC CHECKPOINT ---
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    with open(CHECKPOINT_FILE, "r") as f:
        content = f.read().strip()
        if content.isdigit():
            return int(content)
        return 0

# --- HÀM LƯU CHECKPOINT ---
def save_checkpoint(i):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(i))


# ================================
#   BẮT ĐẦU CHƯƠNG TRÌNH
# ================================
start_index = load_checkpoint()
print(f"▶ Bắt đầu chạy từ i = {start_index}")

# Tạo file CSV nếu chưa có
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])


# Mở CSV ở chế độ append
with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for i in range(start_index, len(test_data)):
        q = test_data[i]

        prompt = f'{q["question"]} Lựa chọn: {q["choices"]}'
        print(f"\n--- Câu {q["qid"]} ---")
        print("User:", prompt)

        try:
            response = rag_service.query(prompt)
            print("Bot:", response)

            answer = str(response).strip().split()[-1]

        except Exception as e:
            print("❌ Error:", e)
            answer = "ERROR"
            break

        # Ghi kết quả
        writer.writerow([q["qid"], answer])
        f.flush()

        # === LƯU CHECKPOINT ===
        save_checkpoint(i + 1)
        

