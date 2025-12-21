#!/usr/bin/env python3
from core import DATABASE_FILE, JSON_OUTPUT, get_logger
from services import EmbeddingsService, Storage

log = get_logger(__name__)


def main():
    log.info("indexing_started")

    storage = Storage(DATABASE_FILE)
    documents = storage.load_documents(JSON_OUTPUT)
    if not documents:
        log.error("no_documents_found", path=str(JSON_OUTPUT))
        return
    storage.save_chunks(documents)
    chunks = storage.load_chunks()

    embeddings = EmbeddingsService()
    collection = embeddings.init_chroma()
    embeddings.index_chunks(chunks, collection)

    log.info("indexing_completed", count=collection.count())


if __name__ == "__main__":
    main()
