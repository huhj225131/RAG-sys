import sqlite3
import json
import time

DB_PATH = "data_processing/rag_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Quản lý trạng thái URL
    c.execute('''CREATE TABLE IF NOT EXISTS visited_urls (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT,
                    status TEXT, -- 'success', 'failed'
                    last_crawled INTEGER
                )''')

    # Lưu trữ nội dung bài viết 
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    url TEXT,
                    title TEXT,
                    content TEXT,
                    metadata JSON,
                    created_at INTEGER,
                    is_embedded BOOLEAN DEFAULT 0 -- Cờ đánh dấu đã embed chưa (CDC pattern)
                )''')
    conn.commit()
    conn.close()

def check_visited(url_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM visited_urls WHERE url_hash = ?", (url_hash,))
    exists = cursor.fetchone()
    conn.close()
    return exists is not None

def mark_visited(url, url_hash, status="success"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO visited_urls (url_hash, url, status, last_crawled)
                      VALUES (?, ?, ?, ?)''', (url_hash, url, status, int(time.time())))
    conn.commit()
    conn.close()

def save_document(doc_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT OR IGNORE INTO documents (id, url, title, content, metadata, created_at)
                          VALUES (?, ?, ?, ?, ?, ?)''', 
                          (doc_data['id'], doc_data['url'], doc_data['title'], 
                           doc_data['text'], json.dumps(doc_data), doc_data['crawled_at']))
        conn.commit()
        return True 
    except Exception as e:
        print(f"Lỗi DB: {e}")
        return False
    finally:
        conn.close()

def get_pending_documents():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE is_embedded = 0")
    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_embedded(doc_ids):
    if not doc_ids: return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholder = '?' + ',?' * (len(doc_ids) - 1) 
    cursor.execute(f"UPDATE documents SET is_embedded = 1 WHERE id IN ({placeholder})", doc_ids)
    conn.commit()
    conn.close()