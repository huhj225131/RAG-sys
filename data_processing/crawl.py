import os, re, json, time, hashlib
import requests
from bs4 import BeautifulSoup
import trafilatura

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-admin-crawler/1.0; +https://example.local)"
}

OUT_DIR = "data_raw/admin"
OUT_JSONL = os.path.join(OUT_DIR, "docs.jsonl")
VISITED = os.path.join(OUT_DIR, "visited.txt")

os.makedirs(OUT_DIR, exist_ok=True)

def norm_url(u: str) -> str:
    u = u.strip()
    u = re.sub(r"#.*$", "", u)
    return u

def url_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def load_visited():
    if not os.path.exists(VISITED):
        return set()
    with open(VISITED, "r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())

def save_visited(visited: set):
    with open(VISITED, "w", encoding="utf-8") as f:
        for u in sorted(visited):
            f.write(u + "\n")

def fetch(url: str, timeout=30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        href = requests.compat.urljoin(base_url, href)
        href = norm_url(href)
        links.append(href)
    return links

def clean_text_from_html(html: str, url: str) -> str:
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return (downloaded or "").strip()

def guess_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    t = soup.title.get_text(" ", strip=True) if soup.title else ""
    return t[:300]

def should_keep(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False

    # bỏ qua các scheme không phải web
    low = url.lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return False

    # bỏ qua file tĩnh / download nặng
    bad_ext = (
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js",
        ".mp4", ".mp3", ".zip", ".rar", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"
    )
    if low.endswith(bad_ext):
        return False

    # chỉ giữ http/https
    try:
        scheme = requests.utils.urlparse(url).scheme.lower()
    except Exception:
        return False
    if scheme not in ("http", "https"):
        return False

    return True

def write_jsonl(obj: dict):
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def crawl(seeds, max_pages=500, sleep_sec=1.0):
    visited = load_visited()
    queue = [norm_url(s) for s in seeds if s.strip()]
    pages = 0

    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        if not should_keep(url):
            visited.add(url)
            continue

        try:
            html = fetch(url)
            text = clean_text_from_html(html, url)
            title = guess_title(html)
            if len(text) >= 300:
                doc = {
                    "id": url_id(url),
                    "url": url,
                    "title": title,
                    "source": re.sub(r"^www\.", "", requests.utils.urlparse(url).netloc),
                    "text": text,
                    "crawled_at": int(time.time()),
                    "tags": ["dia_ly_hanh_chinh"],
                }
                write_jsonl(doc)
                print(f"[OK] {url} ({len(text)} chars)")
            else:
                print(f"[SKIP_SHORT] {url} ({len(text)} chars)")

            links = extract_links(html, url)
            for lk in links:
                if lk not in visited and should_keep(lk):
                    queue.append(lk)

            visited.add(url)
            pages += 1
            if pages % 20 == 0:
                save_visited(visited)

            time.sleep(sleep_sec)

        except Exception as e:
            print(f"[ERR] {url} -> {e}")
            visited.add(url)

    save_visited(visited)
    print("DONE. pages:", pages)

if __name__ == "__main__":
    with open("seeds_admin.txt", "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]
    crawl(seeds, max_pages=500, sleep_sec=1.0)
