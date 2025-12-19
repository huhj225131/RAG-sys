
from core import SimpleRAGService,V2RAGService,LLM_Large,LLM_Small,Embedding
from metric import setup_opik
import logging,time,re,sys
from dotenv import load_dotenv
import os,json,random,csv
from llama_index.core import Settings
load_dotenv()

logging.getLogger("opik").setLevel(logging.ERROR)

setup_opik()
class DualLogger:
    def __init__(self, filename="run_logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect toàn bộ print vào file
sys.stdout = DualLogger("run_logs.txt")
sys.stderr = sys.stdout

Settings.context_window=32000
Settings.num_output=4000
Settings.chunk_size=2048
Settings.chunk_overlap=200
Settings.llm = LLM_Large()
Settings.embed_model =Embedding()


# filename="few_shot.json"
# base_dir = os.getenv("DATA_DIR", "./")
# full_path = os.path.join(base_dir, filename)
# if not os.path.exists(full_path):
#             raise FileNotFoundError(f"Không tìm thấy file: {full_path}")
# with open(full_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
# few_shot = []
# for i in range(2):
#         index = random.randint(0, len(data) -1)
#         few_shot.append((f"{data[index]["question"]}\nLựa chọn: {data[index]["choices"]}",f"Giải thích: {data[index]["explanation"]}\nĐáp án: {data[index]["answer"]}"))
# Settings.llm.few_shot_custom(examples=few_shot, system_instruction="Bạn đang giải câu hỏi trắc nghiệm")


filename="test.json"
base_dir = os.getenv("DATA_DIR", "./data")
full_path = os.path.join(base_dir, filename)
if not os.path.exists(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")
with open(full_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
print("\n--- BẮT ĐẦU HỎI ---")





MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
SWITCH_TO_SMALL_AFTER = int(os.getenv("SWITCH_TO_SMALL_AFTER", "3"))    
TRY_BACK_TO_LARGE_EVERY = int(os.getenv("TRY_BACK_TO_LARGE_EVERY", "20")) 

DATA_DIR = os.getenv("DATA_DIR", "./data")
CHECKPOINT_FILE = "checkpoint.txt"
RESULT_FILE = "result.csv"

def extract_answer(text: str, valid_letters=None) -> str:
    m = re.search(r"Đáp án:\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if not m:
        return "ERROR_FORMAT"

    ans = m.group(1).upper()
    if valid_letters is not None and ans not in valid_letters:
        return "ERROR_OUT_OF_RANGE"
    return ans

def is_policy_block_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("400" in s and "challengecode" in s) or ("không thể trả lời" in s) or ("badrequesterror" in s)

def find_refusal_choice_letter(choices) -> str | None:
    # trả về chữ cái A/B/C/... nếu tìm thấy lựa chọn từ chối
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
        ("rate limit" in s)
    )


def backoff_sleep(retry: int, last_err: Exception):
    s = str(last_err).lower()
    t = min(10 * (2 ** (retry - 1)), MAX_BACKOFF_SECONDS)

    if "401" in s or "unauthorized" in s:
        t = min(max(t, 60), MAX_BACKOFF_SECONDS)  # tối thiểu 60s nếu 401

    print(f"Quota/rate limit → sleep {t}s")
    time.sleep(t)


def load_checkpoint() -> int:
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        s = f.read().strip()
        return int(s) if s.isdigit() else 0

def save_checkpoint(i: int):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(i))

few_shot_path = os.path.join(DATA_DIR, "few_shot.json")
if not os.path.exists(few_shot_path):
    raise FileNotFoundError(f"Không tìm thấy few_shot.json tại: {few_shot_path}")

with open(few_shot_path, "r", encoding="utf-8") as f:
    few_shot_data = json.load(f)

random.seed(42)
few_shot_items = random.sample(few_shot_data, 2)
few_shot = [
    (
        f"Câu hỏi: {item['question']}\nLựa chọn: {item['choices']}",
        f"Giải thích: {item['explanation']}\nĐáp án: {item['answer']}"
    )
    for item in few_shot_items
]

def apply_fewshot():
    Settings.llm.few_shot_custom(
        examples=few_shot,
        system_instruction="Bạn đang giải câu hỏi trắc nghiệm"
    )

Settings.llm = LLM_Large()
apply_fewshot()
rag_service = V2RAGService()
print("Dùng LLM_Large (init)")

current_mode = "large"
rate_limit_streak = 0 # số lần liên tiếp bị rate limit
since_switched = 0 # số câu đã xử lý kể từ khi chuyển LLM

# Functions to switch LLMs
def set_llm_large():
    global current_mode, since_switched, rate_limit_streak
    Settings.llm = LLM_Large()
    apply_fewshot()
    rag_service.rebuild_query_engine()
    current_mode = "large"
    since_switched = 0
    rate_limit_streak = 0
    print("Dùng LLM_Large")


def set_llm_small():
    global current_mode, since_switched, rate_limit_streak
    Settings.llm = LLM_Small()
    apply_fewshot()
    rag_service.rebuild_query_engine()
    current_mode = "small"
    since_switched = 0
    rate_limit_streak = 0
    print("Dùng LLM_Small")

test_path = os.path.join(DATA_DIR, "test.json")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Không tìm thấy test.json tại: {test_path}")

with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

start_index = load_checkpoint()
print(f"▶ Bắt đầu từ i = {start_index} / {len(test_data)}")
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["qid", "answer"])
with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for i in range(start_index, len(test_data)):
        q = test_data[i]
        prompt = f"{q['question']}\nLựa chọn: {q['choices']}"
        print(f"\n--- Câu {q['qid']} ---")
        print(prompt)
        retry = 0
        answer = "ERROR"
        choices = q.get("choices", [])
        valid_letters = {chr(ord("A") + j) for j in range(len(choices))}
        valid_letters.add("X")
        while True:
            try:
                response = rag_service.query(prompt)
                text = str(response)
                print("Bot:", text)

                answer = extract_answer(text, valid_letters=valid_letters)
                if answer == "ERROR_OUT_OF_RANGE":
                    answer = "X"
                if answer == "ERROR_FORMAT":
                    print("Output sai format, raw:", text)
                rate_limit_streak = 0
                break

            except Exception as e:
                print("Error:", e)

                if is_rate_limit_error(e):
                    retry += 1
                    rate_limit_streak += 1

                    # switch Large -> Small nếu 401 liên tiếp
                    if current_mode == "large" and rate_limit_streak >= SWITCH_TO_SMALL_AFTER:
                        set_llm_small()

                    backoff_sleep(retry, e)
                    continue

                if is_policy_block_error(e):
                    letter = find_refusal_choice_letter(choices)
                    answer = letter if letter else "X"
                    print("Policy block → auto choose refusal:", answer)
                    break

                answer = "ERROR"
                break

        writer.writerow([q["qid"], answer])
        f.flush()
        save_checkpoint(i + 1)

        # Try back to large
        if current_mode == "small":
            since_switched += 1
            if since_switched >= TRY_BACK_TO_LARGE_EVERY:
                print("Thử quay lại LLM_Large")
                set_llm_large()
print(f"Số lần sử dụng RAG {rag_service.count_rag}")
print(f"Số lần sử dụng RAG đúng {rag_service.rag_hit}")
print("CHẠY XONG")

