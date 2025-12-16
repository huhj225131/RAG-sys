import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import blobfile as bf, os
import pymupdf4llm
import pathlib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



# The remainder of the script is unchanged
crawl_dir = os.environ.get("DATA_CRAWL", "./crawl")
pdf_dir = Path(crawl_dir) / "pdf"
md_dir  = Path(crawl_dir) / "md"

md_dir.mkdir(parents=True, exist_ok=True)
for name in bf.listdir(str(pdf_dir)):
    if not name.lower().endswith(".pdf"):
        continue

    pdf_path = pdf_dir / name
    md_path  = md_dir / pdf_path.with_suffix(".md").name
    if md_path.is_file():
        continue

    md_text = pymupdf4llm.to_markdown(pdf_path)

    md_path.write_text(md_text, encoding="utf-8")

    print("Converted:", name)







