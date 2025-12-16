import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from core import HEADERS, get_logger, Document

log = get_logger(__name__)


class Crawler:
    def __init__(self, headers: dict = HEADERS):
        self.visited = set()
        self.documents = {}
        self.failed = []
        self.session = requests.Session()
        self.session.headers.update(headers)

    def fetch(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=15)
            return response.text if response.status_code == 200 else None
        except Exception:
            return None

    def extract_text(self, html: str) -> str | None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            return text if len(text) > 100 else None
        except Exception:
            return None

    def extract_title(self, html: str, url: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")

            if title := soup.find("title"):
                return title.get_text().strip()

            if h1 := soup.find("h1"):
                return h1.get_text().strip()

            path = urlparse(url).path
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        except Exception:
            return "Document"

    def extract_links(self, html: str, base_url: str) -> list[str]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            links = []

            for link in soup.find_all("a", href=True):
                href = link["href"].strip()
                if not href or href.startswith("#"):
                    continue

                url = urljoin(base_url, href)
                url = self._normalize_url(url)

                if self._is_valid_url(url):
                    links.append(url)

            return list(set(links))
        except Exception:
            return []

    def crawl(self, seed_urls: list[str], max_pages: int = 200) -> dict[str, Document]:
        queue = list(seed_urls)
        processed = 0

        log.info("crawl_started", max_pages=max_pages, seed_count=len(seed_urls))

        while queue and processed < max_pages:
            url = queue.pop(0)
            url = self._normalize_url(url)

            if url in self.visited or not self._is_valid_url(url):
                continue

            self.visited.add(url)
            processed += 1

            html = self.fetch(url)
            if not html:
                log.warning("fetch_failed", url=url)
                self.failed.append(url)
                time.sleep(1)
                continue

            title = self.extract_title(html, url)
            text = self.extract_text(html)

            if text:
                doc = Document(url=url, title=title, text=text)
                self.documents[url] = doc
                log.info(
                    "page_parsed",
                    url=url,
                    title=title,
                    chars=len(text),
                    progress=f"{processed}/{max_pages}",
                )
            else:
                log.warning("no_text", url=url)
                self.failed.append(url)

            links = self.extract_links(html, url)
            new_links = [
                link for link in links if link not in self.visited and link not in queue
            ]
            queue.extend(new_links)

            if new_links:
                log.debug("links_found", count=len(new_links), queue_size=len(queue))

            time.sleep(0.5)

        log.info(
            "crawl_finished",
            success=len(self.documents),
            failed=len(self.failed),
            visited=len(self.visited),
        )

        return self.documents

    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if "cloud.ru" not in parsed.netloc:
                return False

            path = parsed.path.lower()
            if not path.startswith("/docs/"):
                return False

            skip = ["/search", "/api/", "/static/", ".pdf", ".zip"]
            return not any(pattern in path for pattern in skip)
        except Exception:
            return False

    def _normalize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            path = parsed.path.rstrip("/")
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        except Exception:
            return url
