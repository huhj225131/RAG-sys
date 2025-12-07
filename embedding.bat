@echo off
set PERSIST_DIR=chroma_store
set COLLECTION=emb
set DATA_DIR=crawl

python embedding.py ^
  --persist-dir "%PERSIST_DIR%" ^
  --collection "%COLLECTION%" ^
  --data-dir "%DATA_DIR%" ^
  > log/embed.log 2>&1


