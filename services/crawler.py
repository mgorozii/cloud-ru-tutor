import json
import os
import shutil
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider

import trafilatura

from core import HEADERS, JSON_OUTPUT, Document, get_logger

log = get_logger(__name__)


def _normalize_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}"


class DocsSpider(scrapy.Spider):
    name = "docs"
    custom_settings = {
        "CONCURRENT_REQUESTS": 64,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 16,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 30,
        "ROBOTSTXT_OBEY": False,
        "DEFAULT_REQUEST_HEADERS": HEADERS,
        "LOG_LEVEL": "WARNING",
    }

    def __init__(
        self,
        sources: list[str],
        max_pages: int | None,
        rules: list[dict],
        downloaded: set[str],
        collector: dict[str, Document],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_urls = sources
        self.max_pages = max_pages or float("inf")
        self.rules = rules
        self.downloaded = downloaded
        self.collected = collector
        self.processed = 0
        domains = set()
        for r in rules:
            if r["type"] == "domain":
                domains.add(r["value"].replace("https://", "").replace("http://", ""))
            elif r["type"] == "prefix":
                d = urlparse(r["value"]).netloc
                if d:
                    domains.add(d)
        self.allowed_domains = list(domains)

    async def start(self):
        for url in self.start_urls:
            yield scrapy.Request(url, self.parse, dont_filter=True)

    def parse(self, response):
        if self.processed >= self.max_pages:
            raise CloseSpider("limit reached")

        url = _normalize_url(response.url)
        if url in self.downloaded:
            return

        title, text = self.extract(response)
        if text:
            self.collected[url] = Document(
                url=url, title=title or "Document", text=text
            )
            self.processed += 1
            log.info(
                "page_parsed",
                url=url,
                title=title,
                chars=len(text),
                progress=f"{self.processed}/{int(self.max_pages) if self.max_pages != float('inf') else '?'}",
            )
        else:
            log.info("extract_failed", url=url)

        allowed_count = 0
        for href in response.css("a::attr(href)").getall():
            href = href.strip()
            if not href or href.startswith("#"):
                continue
            next_url = _normalize_url(urljoin(response.url, href))
            if self._allowed(next_url):
                allowed_count += 1
                yield response.follow(next_url, self.parse)

        if allowed_count:
            log.info("links_found", url=url, count=allowed_count)

    def extract(self, response) -> tuple[str | None, str | None]:
        title = response.css("title::text").get() or response.css("h1::text").get()
        text = None

        try:
            text = trafilatura.extract(
                response.text,
                url=response.url,
                include_comments=False,
                include_tables=False,
            )
        except Exception:
            pass

        if not text:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                raw = soup.get_text("\n")
                text = "\n".join(
                    line for line in (s.strip() for s in raw.splitlines()) if line
                )
            except Exception:
                pass

        return (title, text) if text and len(text) >= 100 else (title, None)

    def _allowed(self, url: str) -> bool:
        p = urlparse(url)
        path = p.path.lower()
        if any(
            s in path
            for s in [
                "/search",
                "/api/",
                "/static/",
                ".pdf",
                ".zip",
                ".jpg",
                ".png",
                ".gif",
            ]
        ):
            return False
        for r in self.rules:
            if r["type"] == "domain":
                val = r["value"].replace("https://", "").replace("http://", "")
                if p.netloc == val or p.netloc.endswith(f".{val}"):
                    return True
            elif r["type"] == "prefix" and url.startswith(r["value"]):
                return True
        return False


class Crawler:
    def __init__(self, headers: dict = HEADERS, output_path=None):
        self.documents: dict[str, Document] = {}
        self.failed: list[str] = []
        self.headers = headers
        self.output_path = output_path or JSON_OUTPUT

    def crawl(
        self, seed_urls: list[str] | None = None, max_pages: int | None = None
    ) -> dict[str, Document]:
        sources_env = os.getenv("CRAWL_SOURCES")
        sources = (
            [s.strip() for s in sources_env.split(",")]
            if sources_env
            else (seed_urls or [])
        )
        if not sources:
            log.warning("no_sources")
            return {}

        resume = os.getenv("CRAWL_RESUME", "true").lower() in {"1", "true", "yes"}
        jobdir = os.getenv("CRAWL_JOBDIR", "data/scrapy_job")

        if resume:
            self.documents = self._load_documents()
            downloaded = self._load_downloaded()
            if self.documents:
                log.info("documents_restored", count=len(self.documents))
            os.makedirs(jobdir, exist_ok=True)
        else:
            downloaded = set()
            if os.path.exists(jobdir):
                shutil.rmtree(jobdir, ignore_errors=True)
                log.info("jobs_cleaned")
            if self.output_path.exists():
                self.output_path.unlink()
            jobdir = None

        rules = [self._build_rule(src) for src in sources]
        settings = {"JOBDIR": jobdir} if jobdir else {}
        process = CrawlerProcess(settings=settings)
        process.crawl(
            DocsSpider,
            sources=sources,
            max_pages=max_pages,
            rules=rules,
            downloaded=downloaded,
            collector=self.documents,
        )
        process.start()

        # сохраняем сразу
        # можно работать не скачивая всю документацию
        if self.documents:
            self._save_json()

        log.info("crawl_finished", docs=len(self.documents))
        return self.documents

    def _build_rule(self, src: str) -> dict:
        p = urlparse(src)
        path = (p.path or "/").rstrip("/")
        return (
            {"type": "domain", "value": p.netloc}
            if path in {"", "/"}
            else {"type": "prefix", "value": src.rstrip("/")}
        )

    def _load_downloaded(self) -> set[str]:
        try:
            if self.output_path.exists():
                with open(self.output_path, encoding="utf-8") as f:
                    return set(json.load(f).keys())
        except Exception:
            pass
        return set()

    def _load_documents(self) -> dict[str, Document]:
        try:
            if self.output_path.exists():
                with open(self.output_path, encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    url: Document(
                        url=d.get("url", url),
                        title=d.get("title", "Document"),
                        text=d.get("text", ""),
                    )
                    for url, d in data.items()
                }
        except Exception:
            pass
        return {}

    def _save_json(self):
        try:
            data = {
                url: {"url": doc.url, "title": doc.title, "text": doc.text}
                for url, doc in self.documents.items()
            }
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            size_mb = self.output_path.stat().st_size / 1024 / 1024
            log.info(
                "json_saved",
                path=str(self.output_path),
                docs=len(data),
                size_mb=f"{size_mb:.1f}",
            )
        except Exception as e:
            log.error("json_save_failed", error=str(e))
