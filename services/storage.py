import json
import sqlite3
from pathlib import Path

from core import CHUNK_SIZE, CHUNK_OVERLAP, get_logger, Chunk, Document

log = get_logger(__name__)


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def save_json(self, documents: dict[str, Document], output_path: Path):
        data = {
            url: {"title": doc.title, "text": doc.text, "url": doc.url}
            for url, doc in documents.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        size_mb = output_path.stat().st_size / 1024 / 1024
        log.info(
            "json_saved",
            path=str(output_path),
            docs=len(data),
            size_mb=f"{size_mb:.1f}",
        )

    def save_chunks(self, documents: dict[str, Document]):
        if self.db_path.exists():
            self.db_path.unlink()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                source_url TEXT,
                source_title TEXT,
                chunk_text TEXT
            )
        """)

        chunk_id = 0

        log.info("chunking_started", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for url, doc in documents.items():
            chunks = self._chunk_text(doc.text)

            for chunk_text in chunks:
                cursor.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?)",
                    (f"chunk_{chunk_id}", url, doc.title, chunk_text),
                )
                chunk_id += 1

            log.debug("doc_chunked", title=doc.title, chunks=len(chunks))

        conn.commit()
        conn.close()

        log.info("chunks_saved", path=str(self.db_path), total_chunks=chunk_id)

    def load_chunks(self) -> list[Chunk]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} not found")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT doc_id, source_url, source_title, chunk_text FROM documents"
        )
        rows = cursor.fetchall()
        conn.close()

        chunks = [
            Chunk(id=row[0], source_url=row[1], source_title=row[2], text=row[3])
            for row in rows
        ]

        log.info("chunks_loaded", count=len(chunks))
        return chunks

    def get_source_info(self, doc_id: str) -> dict | None:
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT source_url, source_title FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            result = cursor.fetchone()
            conn.close()

            return {"url": result[0], "title": result[1]} if result else None
        except Exception:
            return None

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP if end < len(text) else len(text)

        return chunks
