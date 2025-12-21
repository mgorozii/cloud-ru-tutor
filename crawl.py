#!/usr/bin/env python3
from core import SEED_URLS, JSON_OUTPUT, DATABASE_FILE, get_logger
from services import Crawler, Storage

log = get_logger(__name__)


def main():
    log.info("crawl_started")

    crawler = Crawler()
    documents = crawler.crawl(SEED_URLS, max_pages=None)

    storage = Storage(DATABASE_FILE)
    storage.save_json(documents, JSON_OUTPUT)

    log.info("pipeline_completed", docs=len(documents), failed=len(crawler.failed))


if __name__ == "__main__":
    main()
