import os, re, json, time, hashlib
import requests
from bs4 import BeautifulSoup
import trafilatura

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-admin-crawler/1.0; +https://example.local)"
}
DEFAULT_OUT_DIR = "data_processing"

def norm_url(u: str) -> str:
    u = u.strip()
    u = re.sub(r"#.*$", "", u)
    return u

def url_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def load_visited(visited_path):
    if not os.path.exists(visited_path):
        return set()
    with open(visited_path, "r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())

def save_visited(visited: set, visited_path):
    os.makedirs(os.path.dirname(visited_path), exist_ok=True)
    with open(visited_path, "w", encoding="utf-8") as f:
        for u in sorted(visited):
            f.write(u + "\n")

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
        if not href: continue
        href = requests.compat.urljoin(base_url, href)
        links.append(norm_url(href))
    return links

def should_keep(url: str) -> bool:
    url = (url or "").strip()
    if not url: return False
    low = url.lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:")): return False
    bad_ext = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".mp4", ".mp3", ".zip", ".pdf", ".docx")
    if low.endswith(bad_ext): return False
    return True

def write_jsonl(obj: dict, out_path: str):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run_crawler(seeds: list, max_pages=500, re_crawl_seeds=True, output_dir=DEFAULT_OUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    out_jsonl = os.path.join(output_dir, "docs.jsonl")
    visited_path = os.path.join(output_dir, "visited.txt")
    
    visited = load_visited(visited_path)
    
    # Xóa seeds khỏi visited tạm thời để crawler quét lại trang chủ tìm link mới
    if re_crawl_seeds:
        for s in seeds:
            norm_s = norm_url(s)
            if norm_s in visited:
                visited.remove(norm_s)

    queue = [norm_url(s) for s in seeds if s.strip()]
    pages = 0
    new_docs_count = 0
    print(f"[Crawler] Bắt đầu với {len(queue)} seeds")
    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        if not should_keep(url):
            visited.add(url)
            continue

        try:
            print(f"Visiting: {url}")
            html = fetch(url)
            text = clean_text_from_html(html, url)
            
            if len(text) >= 300:
                doc = {
                    "id": url_id(url),
                    "url": url,
                    "title": guess_title(html),
                    "source": requests.utils.urlparse(url).netloc,
                    "text": text,
                    "crawled_at": int(time.time()),
                    "tags": ["auto_crawl"],
                }
                write_jsonl(doc, out_jsonl)
                new_docs_count += 1
                print(f"   -> [SAVED] {len(text)} chars")
            
            # Extract links
            links = extract_links(html, url)
            for lk in links:
                if lk not in visited and should_keep(lk):
                    queue.append(lk)

            visited.add(url)
            pages += 1
            if pages % 10 == 0: save_visited(visited, visited_path)
            time.sleep(1.0)

        except Exception as e:
            print(f"   -> [ERR] {e}")
            visited.add(url)

    save_visited(visited, visited_path)
    print(f"--- [Crawler] DONE. New docs: {new_docs_count} ---")
    return out_jsonl, new_docs_count

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