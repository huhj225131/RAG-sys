# Hệ thống RAG cơ bản
## Cài đặt thư viện
```sh
pip install requirements.txt
```

## Sử dụng

```sh
python main.py
```

## Thành phần

### Chroma_store
Lưu dữ liệu vector
### Core
Chứa các lớp override của llamaindex
### Embedding.py
Code thực hiện embedding tạo vector db
### Metric
Config opik để sau tính accuracy
##Log
Log các thành phần chạy (hiện tại có embedding)
