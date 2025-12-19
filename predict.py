
from core import SimpleRAGService,V2RAGService,LLM_Large,LLM_Small,Embedding,V3RAGService
import logging,time,re,sys
from dotenv import load_dotenv
import os,json,random,csv
from llama_index.core import Settings
import string
from utils import extract_answer,is_policy_block_error,find_refusal_choice_letter,is_rate_limit_error,backoff_sleep,load_checkpoint,save_checkpoint,get_few_shot_examples
load_dotenv()


Settings.context_window=32000
Settings.num_output=4000
Settings.chunk_size=2048
Settings.chunk_overlap=200
Settings.llm = LLM_Large()
Settings.embed_model =Embedding()

MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
SWITCH_TO_SMALL_AFTER = int(os.getenv("SWITCH_TO_SMALL_AFTER", "3"))    
TRY_BACK_TO_LARGE_EVERY = int(os.getenv("TRY_BACK_TO_LARGE_EVERY", "20")) 
DATA_DIR = os.getenv("DATA_DIR", "./")
CHECKPOINT_FILE = "checkpoint.txt"
RESULT_FILE = "submission.csv"
TIME_RESULT_FILE = "submission_time.csv"
QUESTION_FILE="test.json"

rag_service = V3RAGService()
print("Dùng LLM_Large (init)")

current_mode = "large"
rate_limit_streak = 0 # số lần liên tiếp bị rate limit
since_switched = 0 # số câu đã xử lý kể từ khi chuyển LLM
# Functions to switch LLMs
def set_llm_large():
    global current_mode, since_switched, rate_limit_streak
    Settings.llm = LLM_Large()
    rag_service.rebuild_query_engine()
    current_mode = "large"
    since_switched = 0
    rate_limit_streak = 0
    print("Dùng LLM_Large")


def set_llm_small():
    global current_mode, since_switched, rate_limit_streak
    Settings.llm = LLM_Small()
    rag_service.rebuild_query_engine()
    current_mode = "small"
    since_switched = 0
    rate_limit_streak = 0
    print("Dùng LLM_Small")


test_path = os.path.join(DATA_DIR, QUESTION_FILE)
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Không tìm thấy test.json tại: {test_path}")

with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("\n--- BẮT ĐẦU HỎI ---")    
#Read checkpoint question
start_index = load_checkpoint()
print(f"▶ Bắt đầu từ i = {start_index} / {len(test_data)}")

#Create submission file
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["qid", "answer"])
if not os.path.exists(TIME_RESULT_FILE):
    with open(TIME_RESULT_FILE,"w",newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["qid","answer","time"])


with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f_res , \
     open(TIME_RESULT_FILE, "a", newline="", encoding="utf-8") as f_time:
    res_writer = csv.writer(f_res)
    time_write = csv.writer(f_time)
    for i in range(start_index, len(test_data)):
        #Start time
        start= time.time()
        q = test_data[i]
        choices = "\n".join(
            f"{letter}.{text}"
            for letter, text in zip(string.ascii_uppercase, q["choices"])
            )
        prompt = f"{q['question']}\nLựa chọn: {choices}"
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
                if is_policy_block_error(e):
                    letter = find_refusal_choice_letter(choices)
                    answer = letter if letter else "X"
                    print("Policy block → auto choose refusal:", answer)
                    break
                print("Error:", e)
                if is_rate_limit_error(e):
                    retry += 1
                    rate_limit_streak += 1

                    # switch Large -> Small nếu 401 liên tiếp
                    if current_mode == "large" and rate_limit_streak >= SWITCH_TO_SMALL_AFTER:
                        set_llm_small()

                    backoff_sleep(retry, e)
                    continue

                answer = "ERROR"
                break
        #End time
        end = time.time()
        res_writer.writerow([q["qid"], answer])
        time_write.writerow([q["qid"],answer,end - start])
        f_res.flush()
        f_time.flush()
        save_checkpoint(i + 1)
        # Try back to large
        if current_mode == "small":
            since_switched += 1
            if since_switched >= TRY_BACK_TO_LARGE_EVERY:
                print("Thử quay lại LLM_Large")
                set_llm_large()
print("CHẠY XONG")

