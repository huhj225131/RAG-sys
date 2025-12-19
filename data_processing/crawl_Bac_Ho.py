import os, re, json, time, hashlib
from urllib.parse import urlparse

import requests
import trafilatura

URLS = [
    "https://bvhttdl.gov.vn/cuoc-doi-va-su-nghiep-cach-mang-ve-vang-cua-chu-tich-ho-chi-minh-20201026145330541.htm",
    "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ch%C3%AD_Minh",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw", "bac_ho_2sites")
OUT_JSONL = os.path.join(OUT_DIR, "docs.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

SESSION = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def norm_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"#.*$", "", u)
    if not u:
        return ""
    p = urlparse(u)
    if not p.scheme:
        u = "https://" + u.lstrip("/")
    return u

def url_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def fetch_html(url: str, timeout=25) -> str:
    r = SESSION.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    # fix encoding nếu server trả sai
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text

def extract_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True
    )
    return (text or "").strip()

def extract_title(html: str) -> str:
    # dùng trafilatura metadata cho ổn (Wikipedia title hay sạch hơn)
    md = trafilatura.extract_metadata(html)
    if md and md.title:
        return md.title.strip()[:300]
    return ""

def write_jsonl(obj: dict):
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def crawl_fixed_urls(urls: list[str]):
    # clear output mỗi lần chạy (nếu bạn muốn append thì bỏ 2 dòng này)
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)

    for u in urls:
        url = norm_url(u)
        try:
            html = fetch_html(url)
            text = extract_text(html, url)
            title = extract_title(html) or ""

            if len(text) < 200:
                print(f"[SKIP_SHORT] {url} ({len(text)} chars)")
                continue

            host = re.sub(r"^www\.", "", urlparse(url).netloc.lower())

            doc = {
                "id": url_id(url),
                "url": url,
                "title": title,
                "source": host,
                "text": text,
                "crawled_at": int(time.time()),
                "tags": ["Bac_Ho", host],
            }
            write_jsonl(doc)
            print(f"[OK] {host} ({len(text)} chars) -> {url}")

        except Exception as e:
            print(f"[ERR] {url} -> {e}")

    print("DONE. Output:", OUT_JSONL)

if __name__ == "__main__":
    crawl_fixed_urls(URLS)
