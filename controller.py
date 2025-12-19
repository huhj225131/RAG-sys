# pipeline_controller.py
import os
import shutil
from data_processing.crawl import run_crawler
from data_processing.embedding import embed_jsonl_file
from data_processing.embed_word import process_docx_list
from data_processing.embed_md import process_md_list

SEEDS_FILE = "data_processing/seeds_admin.txt"
CHROMA_DIR = "./chroma_store"
DOCSTORE_DIR = "./docstore_save"
UPLOAD_TEMP_DIR = "./temp_uploads"

def update_web_data():
    if not os.path.exists(SEEDS_FILE):
        return False, "Không tìm thấy file seeds!"
    
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]

    jsonl_path, count = run_crawler(seeds, max_pages=50, re_crawl_seeds=True)
    
    if count > 0:
        embed_count = embed_jsonl_file(jsonl_path, persist_dir=CHROMA_DIR)
        return True, f"Cập nhật thành công {embed_count} bài viết mới."
    else:
        return True, "Hệ thống đã cập nhật. Không có bài mới."

def process_uploaded_files(uploaded_streamlit_files):
    if not os.path.exists(UPLOAD_TEMP_DIR):
        os.makedirs(UPLOAD_TEMP_DIR)
        
    docx_paths = []
    md_paths = []
    
    saved_paths = []
    for up_file in uploaded_streamlit_files:
        file_path = os.path.join(UPLOAD_TEMP_DIR, up_file.name)
        with open(file_path, "wb") as f:
            f.write(up_file.getbuffer())
        saved_paths.append(file_path)
        
        if file_path.endswith(".docx"):
            docx_paths.append(file_path)
        elif file_path.endswith(".md"):
            md_paths.append(file_path)
            
    total_docs = 0
    status_msg = []
    
    if docx_paths:
        c = process_docx_list(docx_paths, persist_dir=CHROMA_DIR, docstore_dir=DOCSTORE_DIR)
        total_docs += c
        status_msg.append(f"Word: {c} files")
        
    if md_paths:
        c = process_md_list(md_paths, persist_dir=CHROMA_DIR, docstore_dir=DOCSTORE_DIR)
        total_docs += c
        status_msg.append(f"Markdown: {c} files")

    for p in saved_paths:
        try: os.remove(p)
        except: pass
        
    return f"Hoàn tất! Tổng cộng {total_docs} documents. ({', '.join(status_msg)})"