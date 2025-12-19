# Hệ thống RAG 
## 1. Luồng xử lý dữ liệu (Pipeline Flow)

Quy trình xử lý một câu hỏi đi qua các bước tuần tự như sau:

### Bước 1: Tiếp nhận và Đánh giá ban đầu (Giai đoạn 1)
* Hệ thống nhận **Câu hỏi đầu vào**.
* Câu hỏi được đưa vào module **Đánh giá trực tiếp** sử dụng Prompt quy tắc và các mẫu Few-shot.
* LLM thực hiện đánh giá sơ bộ. Tại đây, hệ thống kiểm tra hai điều kiện:
    1.  LLM có tự tin trả lời ngay không?
    2.  Lĩnh vực của câu hỏi có cần thông tin thêm không
* **Nhánh 1 (Thành công):** Nếu cả hai điều kiện thỏa mãn, hệ thống trả về **Kết quả trực tiếp** ngay lập tức và kết thúc quy trình.

### Bước 2: Truy xuất dữ liệu nâng cao (Giai đoạn 2)
* **Nhánh 2 (Cần tra cứu):** Nếu LLM không chắc chắn hoặc sai định dạng ở Bước 1, câu hỏi được chuyển sang quy trình **RAG nâng cao**.
* Hệ thống thực hiện truy vấn vào **Vector DB** để tìm kiếm các đoạn văn bản cơ sở (**Leaf Nodes**).

### Bước 3: Xử lý hợp nhất ngữ cảnh (Auto-Merging)
* Các *Leaf Nodes* tìm được sẽ đi qua bộ **AutoMergingRetriever**.
* Hệ thống kiểm tra điều kiện hợp nhất:
    * **Trường hợp A:** Nếu tìm thấy đủ số lượng node con liên quan, hệ thống tự động gộp chúng lại thành **Node Cha (Parent Node)** để lấy ngữ cảnh rộng hơn.
    * **Trường hợp B:** Nếu không đủ điều kiện, hệ thống giữ nguyên các **Leaf Nodes** ban đầu.

### Bước 4: Lọc nhiễu và Tổng hợp
* Kết quả từ bước 3 (Parent hoặc Leaf nodes) được đưa qua bộ lọc **Similarity Post-processor**.
* Chỉ những nội dung có điểm tương đồng (score) **> 0.4** mới được giữ lại làm **Ngữ cảnh cuối cùng**.
* Cuối cùng, ngữ cảnh này được đưa vào **LLM Synthesizer** để tổng hợp và trả về **Kết quả RAG**.
<img src="asset/asset1.png" alt="Minh họa luồng làm việc RAG">

---



