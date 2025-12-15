import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf

import pymupdf4llm



# The remainder of the script is unchanged

md_text = pymupdf4llm.to_markdown(r"D:\vnpt_hackathon\crawl\nhasachmienphi-giao-trinh-tu-tuong-ho-chi-minh.pdf")



# now work with the markdown text, e.g. store as a UTF8-encoded file

import pathlib

pathlib.Path("output.md").write_bytes(md_text.encode())