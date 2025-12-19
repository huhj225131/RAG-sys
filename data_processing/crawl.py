import os, re, json, time, hashlib
import requests
from bs4 import BeautifulSoup
import trafilatura
from data_processing.db_manager import init_db, check_visited, mark_visited, save_document, DB_PATH

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-admin-crawler/1.0; +https://example.local)"
}

def norm_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"#.*$", "", u)
    return u

def url_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def fetch(url: str, timeout=30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def clean_text_from_html(html: str, url: str) -> str:
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return (downloaded or "").strip()

def guess_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    t = soup.title.get_text(" ", strip=True) if soup.title else ""
    return t[:300]

def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        href = requests.compat.urljoin(base_url, href)
        links.append(norm_url(href))
    return links

def should_keep(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    low = url.lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return False
    bad_ext = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".mp4", ".mp3", ".zip", ".pdf", ".docx")
    if low.endswith(bad_ext):
        return False
    return True

def run_crawler(seeds: list, max_pages=500, re_crawl_seeds=True):
    init_db()

    seeds_norm = [norm_url(s) for s in seeds if (s or "").strip()]
    seeds_ids = set(url_id(s) for s in seeds_norm)

    queue = list(seeds_norm)
    pages = 0
    new_docs_count = 0

    print(f"[Crawler] Bắt đầu với {len(queue)} seeds")
    while queue and pages < max_pages:
        url = queue.pop(0)
        uid = url_id(url)
        if check_visited(uid) and not (re_crawl_seeds and uid in seeds_ids):
            continue

        if not should_keep(url):
            mark_visited(url, uid, status="skipped")
            continue

        try:
            print(f"Visiting: {url}")
            html = fetch(url)
            text = clean_text_from_html(html, url)

            if len(text) >= 300:
                doc = {
                    "id": uid,
                    "url": url,
                    "title": guess_title(html),
                    "source": requests.utils.urlparse(url).netloc,
                    "text": text,
                    "crawled_at": int(time.time()),
                    "tags": ["auto_crawl"],
                }
                if save_document(doc):
                    new_docs_count += 1
                    print(f"   -> [SAVED] {len(text)} chars")

            links = extract_links(html, url)
            for lk in links:
                lk = norm_url(lk)
                if not lk:
                    continue
                lk_id = url_id(lk)
                if not check_visited(lk_id) and should_keep(lk):
                    queue.append(lk)

            mark_visited(url, uid, status="success")
            pages += 1
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERR] {e}")
            mark_visited(url, uid, status="failed")

    print(f"[Crawler] DONE. New docs: {new_docs_count}. DB: {DB_PATH}")
    return DB_PATH, new_docs_count


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_path = os.path.join(current_dir, "seeds_admin.txt")

    if os.path.exists(seeds_path):
        print(f"Đang chạy TEST chế độ thủ công với file: {seeds_path}")
        with open(seeds_path, "r", encoding="utf-8") as f:
            real_seeds = [line.strip() for line in f if line.strip()]
        run_crawler(real_seeds, max_pages=50, re_crawl_seeds=True)
    else:
        print("Không tìm thấy seeds_admin.txt, chạy demo vnexpress...")
        run_crawler(["https://vnexpress.net"], max_pages=5)
