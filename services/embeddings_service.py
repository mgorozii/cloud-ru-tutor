import time
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from core import CHROMA_DB_DIR, EMBEDDING_MODEL, get_logger, Chunk

log = get_logger(__name__)


class EmbeddingsService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info("loading_model", model=model_name)
        
        try:
            self.model = SentenceTransformer(model_name, device="cpu")
            log.info("model_loaded_cpu")
        except Exception as e:
            log.error("model_loading_failed", error=str(e))
            log.info("trying_simple_load")
            self.model = SentenceTransformer(model_name)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        log.info("model_loaded", model=model_name, embedding_dim=self.embedding_dim)

    def get_chroma_client(self):
        """Создание клиента ChromaDB с правильными настройками"""
        return chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def init_chroma(self) -> chromadb.Collection:
        """Создание новой коллекции"""
        client = self.get_chroma_client()
        
        # Удаляем старую коллекцию если существует
        try:
            client.delete_collection("cloud_docs")
            log.info("old_collection_deleted")
        except Exception as e:
            log.info("no_old_collection_or_error", error=str(e))
        
        # Создаем новую коллекцию
        collection = client.create_collection(
            name="cloud_docs",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 16}
        )
        
        log.info("collection_created")
        return collection

    def load_chroma(self, force_recreate: bool = False) -> chromadb.Collection:
        """Загрузка или создание коллекции"""
        client = self.get_chroma_client()
        
        if force_recreate:
            log.info("forcing_collection_recreation")
            return self.init_chroma()
        
        try:
            collection = client.get_collection("cloud_docs")
            # Проверяем, что коллекция работает
            _ = collection.count()
            log.info("collection_loaded", count=collection.count())
            return collection
        except Exception as e:
            log.warning("collection_load_failed", error=str(e))
            log.info("creating_new_collection")
            return self.init_chroma()

    def index_chunks(self, chunks: List[Chunk], collection: chromadb.Collection, batch_size: int = 32):
        """Индексация чанков"""
        total = len(chunks)
        log.info("indexing_started", total=total)
        
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Создаем уникальный ID
            if len(chunk.text) < 300:
                continue
            
            chunk_hash = hashlib.md5(chunk.text.encode()).hexdigest()[:16]
            chunk_id = f"chunk_{i}_{chunk_hash}"
            
            ids.append(chunk_id)
            texts.append(chunk.text)
            metadatas.append({
                "url": chunk.source_url,
                "title": chunk.source_title,
            })
            
            # Добавляем батчами
            if len(ids) >= batch_size or i == total - 1:
                try:
                    embeddings = self.model.encode(texts)
                    collection.add(
                        ids=ids,
                        embeddings=embeddings.tolist(),
                        documents=texts,
                        metadatas=metadatas
                    )
                    log.info("batch_indexed", size=len(ids), total_progress=f"{i+1}/{total}")
                    
                    # Очищаем списки
                    ids.clear()
                    texts.clear()
                    metadatas.clear()
                    
                except Exception as e:
                    log.error("batch_indexing_failed", error=str(e))
                    # Пробуем с меньшим батчем
                    if batch_size > 8:
                        return self.index_chunks(chunks, collection, batch_size // 2)
        
        log.info("indexing_completed", total_indexed=collection.count())

    def search(self, query: str, collection: chromadb.Collection, top_k: int = 10) -> List[dict]:
        """Поиск с обработкой ошибок"""
        try:
            query_embedding = self.model.encode(query)
            query = query + " kubernetes кластер сервис подключение"
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                log.warning("no_results_found", query=query[:50])
                return []
            
            docs = []
            for i in range(len(results["ids"][0])):
                docs.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]
                })
            
            log.info("search_completed", found=len(docs), query=query[:50])
            return docs
            
        except Exception as e:
            log.error("search_error", query=query[:50], error=str(e))
            return []

    def get_collection_info(self, collection: Optional[chromadb.Collection] = None) -> dict:
        """Получение информации о коллекции"""
        try:
            if collection is None:
                client = self.get_chroma_client()
                collection = client.get_collection("cloud_docs")
            
            count = collection.count()
            return {
                "approximate_count": count,
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            log.error("collection_info_error", error=str(e))
            return {"approximate_count": 0, "embedding_dim": self.embedding_dim}