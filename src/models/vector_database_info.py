import json
from typing import List, Dict

from typing import override

from src.enums.embedding_type import EmbeddingType


class VectorDatabaseInfo:
    supported_embeddings: List[EmbeddingType] = []
    simultaneous_embeddings: int = 0

    def __init__(self, *, embedding_model_name: str, embedding_types: List[EmbeddingType], chunk_size: int, chunk_overlap: int, file_names: List):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_names = file_names
        self.embedding_types=embedding_types

    def metadata_to_dict(self) -> Dict:
        """Konwertuje obiekt do słownika (dict)."""
        return {
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "file_names": self.file_names,
            "embedding_types": self.embedding_types
        }

    @classmethod
    def metadata_from_dict(cls, data: Dict):
        """Tworzy obiekt klasy na podstawie słownika (dict)."""
        return cls(
            embedding_model_name=data.get("embedding_model_name", "N/A"),
            chunk_size=data.get("chunk_size", 0),
            chunk_overlap=data.get("chunk_overlap", 0),
            file_names=data.get("file_names", []),
            embedding_types=data.get("embedding_types", []),
        )

    @property
    def file_count(self) -> int:
        """Getter zwracający liczbę plików w file_names."""
        return len(self.file_names)


class ChromaVectorDatabase(VectorDatabaseInfo):
    supported_embeddings = [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT]
    simultaneous_embeddings: int = 1

    def create_metadata_for_database(self):
        metadata_dict = self.metadata_to_dict()
        metadata_dict["file_names"] = ", ".join(metadata_dict["file_names"])




class LanceVectorDatabase(VectorDatabaseInfo):
    supported_embeddings = [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT]
    simultaneous_embeddings: int = 3