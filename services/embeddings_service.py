import time

import chromadb
from sentence_transformers import SentenceTransformer

from core import CHROMA_DB_DIR, EMBEDDING_MODEL, get_logger, Chunk

log = get_logger(__name__)


class EmbeddingsService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info("loading_model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        log.info(
            "model_loaded",
            model=model_name,
            embedding_dim=self.embedding_dim,
        )

    def init_chroma(self) -> chromadb.Collection:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        try:
            client.delete_collection("cloud_docs")
            log.info("old_collection_deleted")
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name="cloud_docs", metadata={"hnsw:space": "cosine"}
        )

        log.info("collection_created", name="cloud_docs")
        return collection

    def load_chroma(self) -> chromadb.Collection:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("cloud_docs")
        log.info("collection_loaded", name="cloud_docs", count=collection.count())
        return collection

    def index_chunks(
        self, chunks: list[Chunk], collection: chromadb.Collection, batch_size: int = 32
    ):
        total = len(chunks)
        start_time = time.time()

        log.info("indexing_started", total=total, batch_size=batch_size)

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [c.id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "chunk_id": c.id,
                }
                for c in batch
            ]

            embeddings = self.model.encode(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

            elapsed = time.time() - start_time
            processed = min(i + batch_size, total)
            speed = processed / elapsed if elapsed > 0 else 0

            log.info(
                "batch_indexed",
                batch=f"{i // batch_size + 1}/{(total + batch_size - 1) // batch_size}",
                processed=f"{processed}/{total}",
                speed=f"{speed:.0f} chunks/s",
            )

        elapsed = time.time() - start_time
        log.info(
            "indexing_finished",
            total=total,
            time=f"{elapsed:.1f}s",
            speed=f"{total / elapsed:.0f} chunks/s",
        )

    def search(
        self, query: str, collection: chromadb.Collection, top_k: int = 10
    ) -> list:
        query_embedding = self.model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                }
            )

        return docs
