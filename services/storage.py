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
                chunk_index INTEGER,
                chunk_text TEXT
            )
        """)

        chunk_id = 0

        log.info("chunking_started", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for url, doc in documents.items():
            chunks = self._chunk_text(doc.text)

            for index, chunk_text in enumerate(chunks):
                cursor.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (f"chunk_{chunk_id}", url, doc.title, index, chunk_text),
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
            "SELECT doc_id, source_url, source_title, chunk_index, chunk_text FROM documents"
        )
        rows = cursor.fetchall()
        conn.close()

        chunks = [
            Chunk(
                id=row[0],
                source_url=row[1],
                source_title=row[2],
                chunk_index=row[3],
                text=row[4],
            )
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

    def load_documents(self, input_path: Path) -> dict[str, Document]:
        try:
            with open(input_path, encoding="utf-8") as f:
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
            return {}

    def get_chunk_with_neighbors(
        self, doc_id: str, neighbor_count: int = 1, trim_overlap: bool = True
    ) -> list[Chunk]:
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, source_url, source_title, chunk_index, chunk_text "
                "FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return []

            base = Chunk(
                id=row[0],
                source_url=row[1],
                source_title=row[2],
                chunk_index=row[3],
                text=row[4],
            )

            neighbors: list[Chunk] = []
            if neighbor_count > 0:
                indices = [
                    base.chunk_index + offset
                    for offset in range(-neighbor_count, neighbor_count + 1)
                    if offset != 0
                ]
                if indices:
                    cursor.execute(
                        "SELECT doc_id, source_url, source_title, chunk_index, chunk_text "
                        "FROM documents WHERE source_url = ? AND chunk_index IN ({}) "
                        "ORDER BY chunk_index ASC".format(",".join("?" * len(indices))),
                        (base.source_url, *indices),
                    )
                    rows = cursor.fetchall()
                    for nrow in rows:
                        chunk = Chunk(
                            id=nrow[0],
                            source_url=nrow[1],
                            source_title=nrow[2],
                            chunk_index=nrow[3],
                            text=nrow[4],
                        )
                        if trim_overlap:
                            chunk = self._trim_neighbor_overlap(base, chunk)
                        neighbors.append(chunk)

            conn.close()
            previous = [c for c in neighbors if c.chunk_index < base.chunk_index]
            following = [c for c in neighbors if c.chunk_index > base.chunk_index]
            return previous + [base] + following
        except Exception:
            return []

    def _trim_neighbor_overlap(self, base: Chunk, neighbor: Chunk) -> Chunk:
        if CHUNK_OVERLAP <= 0:
            return neighbor
        if neighbor.chunk_index < base.chunk_index:
            trimmed = neighbor.text[:-CHUNK_OVERLAP]
        else:
            trimmed = neighbor.text[CHUNK_OVERLAP:]
        if not trimmed:
            trimmed = neighbor.text
        return Chunk(
            id=neighbor.id,
            source_url=neighbor.source_url,
            source_title=neighbor.source_title,
            chunk_index=neighbor.chunk_index,
            text=trimmed,
        )

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP if end < len(text) else len(text)

        return chunks
