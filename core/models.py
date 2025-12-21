from dataclasses import dataclass


@dataclass
class Document:
    url: str
    title: str
    text: str


@dataclass
class Chunk:
    id: str
    source_url: str
    source_title: str
    chunk_index: int
    text: str


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    distance: float

    @property
    def similarity(self) -> float:
        return 1 - self.distance
