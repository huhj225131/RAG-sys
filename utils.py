import re
import time
import os
import json
import random
import string

# Các hằng số mặc định (có thể override nếu cần)
CHECKPOINT_FILE = "checkpoint.txt"
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "60"))

def extract_answer(text: str, valid_letters=None) -> str:
    # Tìm pattern "Đáp án: A"
    m = re.search(r"Đáp án:\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if not m:
        # Fallback: Thử tìm ký tự A-D đứng đầu câu hoặc cuối câu nếu pattern trên fail
        # (Bạn có thể thêm logic fallback ở đây nếu muốn "liều" hơn)
        return "ERROR_FORMAT"

    ans = m.group(1).upper()
    if valid_letters is not None and ans not in valid_letters:
        return "ERROR_OUT_OF_RANGE"
    return ans

def is_policy_block_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("400" in s and "challengecode" in s) or \
           ("không thể trả lời" in s) or \
           ("badrequesterror" in s)

def find_refusal_choice_letter(choices) -> str | None:
    keywords = ["không thể trả lời", "tôi không thể", "từ chối", "refuse", "cannot answer"]
    for idx, c in enumerate(choices or []):
        t = str(c).lower()
        if any(k in t for k in keywords):
            return chr(ord("A") + idx)
    return None

def is_rate_limit_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        ("401" in s) or ("unauthorized" in s) or
        ("429" in s) or ("too many requests" in s) or
        ("rate limit" in s) or ("quota" in s)
    )

def backoff_sleep(retry: int, last_err: Exception):
    s = str(last_err).lower()
    # Tính thời gian chờ mũ (exponential backoff)
    t = min(5 * (2 ** (retry - 1)), MAX_BACKOFF_SECONDS)

    if "401" in s or "unauthorized" in s:
        # Nếu lỗi xác thực, chờ lâu hơn chút nhưng đừng quá lâu
        t = min(max(t, 60), MAX_BACKOFF_SECONDS) 

    print(f"Warning: Gặp lỗi '{str(last_err)[:50]}...' -> Ngủ {t}s")
    time.sleep(t)

def load_checkpoint() -> int:
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            s = f.read().strip()
            return int(s) if s.isdigit() else 0
    except:
        return 0

def save_checkpoint(i: int):
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            f.write(str(i))
    except:
        pass

def get_few_shot_examples(data_dir="."):
    """
    Hàm load few-shot an toàn.
    Trả về list examples để nạp vào Settings.llm sau này.
    """
    # Xử lý đường dẫn linh hoạt
    filename = "few_shot.json"
    full_path = os.path.join(data_dir, filename)
    
    # Nếu không tìm thấy ở data_dir, thử tìm ở thư mục hiện tại
    if not os.path.exists(full_path):
        full_path = filename
        
    if not os.path.exists(full_path):
        print(f"Warning: Không tìm thấy {filename}, chạy chế độ Zero-shot.")
        return []

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)
        
        random.seed(42)
        # Lấy tối đa 2 mẫu hoặc ít hơn nếu file không đủ
        count = min(2, len(few_shot_data))
        few_shot_items = random.sample(few_shot_data, count)
        
        few_shot_examples = []
        for item in few_shot_items:
            choices = "\n".join(
                f"{letter}.{text}"
                for letter, text in zip(string.ascii_uppercase, item["choices"])
            )
            few_shot_examples.append({
                "query": f"Câu hỏi: {item['question']}\nLựa chọn: {choices}",
                "answer": f"Giải thích: {item['explanation']}\nĐáp án: {item['answer']}"
            })
        return few_shot_examples
    except Exception as e:
        print(f"Error loading few-shot: {e}")
        return []