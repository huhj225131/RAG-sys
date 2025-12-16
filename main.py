
from core import SimpleRAGService,LLM_Large,LLM_Small,Embedding
from metric import setup_opik
import logging
from dotenv import load_dotenv
import os,json,random,csv,string
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
llm_small = LLM_Small()
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
base_dir = os.getenv("DATA_DIR", "./")
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



def classify_question(question):
    """Phân loại câu hỏi: MATH, SENSITIVE, hay RAG"""
    prompt = f"""
    Phân loại câu hỏi sau vào 1 trong 3 nhóm:
    - MATH: Toán học, tính toán, tư duy logic, đố vui.
    - SENSITIVE: Chính trị nhạy cảm, bạo lực, xúc phạm, phi pháp, vi phạm đạo đức.
    - RAG: Các câu hỏi kiến thức Lịch sử, Địa lý, Xã hội cần tra cứu.
    
    Câu hỏi: {question}
    Chỉ trả về đúng tên nhóm (MATH/SENSITIVE/RAG):
    """
    try:
        res = llm_small.complete(prompt)
        ans = res.text.strip().upper() 
        if "MATH" in ans: return "MATH"
        if "SENSITIVE" in ans: return "SENSITIVE"
        return "RAG"
    except:
        return "RAG" 
    
def format_choices(choices_list):
    """
    Biến đổi list ['X', 'Y', 'Z'] thành chuỗi:
    A. X
    B. Y
    C. Z
    ...
    Để gửi vào prompt cho Model dễ chọn.
    """
    formatted_text = ""
    labels = list(string.ascii_uppercase) 
    
    for i, choice in enumerate(choices_list):
        if i < len(labels):
            label = labels[i]
            formatted_text += f"{label}. {choice}\n"
            
    return formatted_text



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



with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for i in range(start_index, len(test_data)):
        q = test_data[i]
        
        question_text = q["question"]
        formatted_choices = format_choices(q["choices"]) 
        
        prompt_for_model = f"""
        Câu hỏi: {question_text}
        Các lựa chọn:
        {formatted_choices}
        """
        
        print(f"\n--- Câu {q.get('qid', i)} ---")
        
        category = classify_question(question_text)
        print(f"Router: [{category}]")
        
        answer = "A" 
        try:
            if category == "SENSITIVE":
                print(">>> Xử lý Sensitive...")
                safety_prompt = f"""
                Bạn là bộ lọc an toàn. Tìm trong các lựa chọn sau đáp án nào mang ý nghĩa TỪ CHỐI TRẢ LỜI hoặc VI PHẠM.
                
                {formatted_choices}
                
                Chỉ trả về 1 ký tự (A, B, C...).
                Đáp án:
                """
                res = llm_small.complete(safety_prompt)
                answer = res.text.strip().split()[-1].replace(".", "")

            elif category == "MATH":
                print(">>> Xử lý Math...")
                stem_prompt = f"""
                Giải bài toán sau từng bước (Step-by-step).
                
                Câu hỏi: {question_text}
                Lựa chọn:
                {formatted_choices}
                
                Hãy tính toán ra nháp. Sau đó chọn đáp án đúng nhất.
                BẮT BUỘC kết thúc bằng dòng: "Đáp án: X" (X là ký tự A, B, C...).
                """
                response = Settings.llm.complete(stem_prompt)
                answer = response.text.strip().split()[-1].replace(".", "")

            else:
                print(">>> Xử lý RAG...")
                response = rag_service.query(prompt_for_model)
                answer = str(response).strip().split()[-1].replace(".", "")

        except Exception as e:
            print(f"❌ Lỗi câu {q.get('qid')}: {e}")
            answer = "A"

        print(f"-> Chốt đáp án: {answer}")

        writer.writerow([q.get("qid"), answer])
        f.flush()

        save_checkpoint(i + 1)