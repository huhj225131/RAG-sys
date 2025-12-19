import os
from pathlib import Path
from dotenv import load_dotenv
import pymupdf.layout 
import pymupdf4llm

load_dotenv()



crawl_dir = os.environ.get("DATA_CRAWL", "./crawl")
pdf_dir = Path(crawl_dir) / "pdf"
md_dir  = Path(crawl_dir) / "md"

md_dir.mkdir(parents=True, exist_ok=True)

# Lấy danh sách file PDF
pdf_files = list(pdf_dir.glob("*.pdf"))
print(f"Tìm thấy {len(pdf_files)} file PDF cần xử lý.")

for pdf_path in pdf_files:
    md_path = md_dir / pdf_path.with_suffix(".md").name

    if md_path.exists():
        print(f"⏩ Bỏ qua (đã tồn tại): {pdf_path.name}")
        continue

    
    try:
        # force_ocr=True: Bắt buộc dùng OCR quét ảnh
        # Lưu ý: Tốc độ sẽ chậm hơn nhiều so với bình thường
        md_text = pymupdf4llm.to_markdown(pdf_path,
                                        force_ocr=False,
                                        language="vie")
        
        md_path.write_text(md_text, encoding="utf-8")
        print(f"✅ Hoàn thành: {pdf_path.name}")
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {pdf_path.name}: {e}")

print("\n--- Xử lý hoàn tất ---")