import json
import csv
import os

# --- CẤU HÌNH TÊN FILE ---
VAL_FILE = "./data/val.json"       # File chứa đáp án đúng
RESULT_FILE = "result2.csv" # File kết quả model chạy ra
LOG_WRONG_FILE = "wrong_answers.csv" # File ghi lại các câu sai để debug

def evaluate():
    # 1. Load đáp án đúng từ val.json
    if not os.path.exists(VAL_FILE):
        print(f"❌ Không tìm thấy file {VAL_FILE}")
        return

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Tạo dictionary: { "val_001": "A", "val_002": "C", ... }
    ground_truth = {item["qid"]: item["answer"].strip().upper() for item in val_data}
    total_questions = len(ground_truth)
    print(f"📂 Đã load {total_questions} câu hỏi từ {VAL_FILE}")

    # 2. Load đáp án của Model từ CSV
    if not os.path.exists(RESULT_FILE):
        print(f"❌ Không tìm thấy file {RESULT_FILE}")
        return

    model_preds = {}
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Lấy qid và answer, chuẩn hóa về chữ in hoa
            if "qid" in row and "answer" in row:
                model_preds[row["qid"]] = row["answer"].strip().upper()
    
    print(f"📂 Đã load {len(model_preds)} câu trả lời từ {RESULT_FILE}")

    # 3. So sánh và chấm điểm
    correct_count = 0
    wrong_cases = [] # Lưu lại câu sai để soi
    missing_count = 0

    for qid, true_ans in ground_truth.items():
        if qid not in model_preds:
            missing_count += 1
            # print(f"⚠️ Thiếu câu {qid} trong file kết quả")
            continue
        
        pred_ans = model_preds[qid]
        
        # Logic so sánh
        if pred_ans == true_ans:
            correct_count += 1
        else:
            wrong_cases.append({
                "qid": qid,
                "truth": true_ans,
                "pred": pred_ans
            })

    # 4. In kết quả
    score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    print("\n" + "="*30)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ")
    print("="*30)
    print(f"✅ Số câu đúng:   {correct_count} / {total_questions}")
    print(f"❌ Số câu sai:    {len(wrong_cases)}")
    print(f"⚠️ Số câu thiếu:  {missing_count}")
    print(f"🎯 ĐỘ CHÍNH XÁC:  {score:.2f}%")
    print("="*30)

    # 5. Ghi file log các câu sai (để bạn biết model đang ngu ở đâu)
    if wrong_cases:
        with open(LOG_WRONG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "truth", "pred"])
            writer.writeheader()
            writer.writerows(wrong_cases)
        print(f"📝 Đã lưu danh sách câu sai vào '{LOG_WRONG_FILE}'. Mở ra xem để sửa Prompt nhé!")

if __name__ == "__main__":
    evaluate()