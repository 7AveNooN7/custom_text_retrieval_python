from typing import List
from src.enums.embedding_type_enum import EmbeddingType

class DownloadedModelInfo:
    """Reprezentuje pobrany model z katalogu."""

    def __init__(self, *, model_name: str, embedding_types: List[EmbeddingType]):
        self.model_name = model_name  # Nazwa modelu
        self.embedding_types = embedding_types

    @classmethod
    def from_json(cls, *, json_data: dict):
        """Tworzy obiekt DownloadedModelInfo na podstawie danych JSON."""
        model_name = json_data.get("model_name")
        embedding_types_raw = json_data.get("embedding_types", [])

        # Konwersja stringów na enumy
        try:
            embedding_types = [EmbeddingType(et) for et in embedding_types_raw]
        except ValueError as e:
            raise ValueError(f"Invalid embedding type in JSON: {e}")

        return cls(model_name=model_name, embedding_types=embedding_types)

    def to_json(self) -> dict:
        """Zwraca obiekt w formacie JSON."""
        return {
            "model_name": self.model_name,
            "embedding_types": [et.value for et in self.embedding_types]  # Konwersja enuma na stringi
        }

    def __repr__(self):
        return f"DownloadedModelInfo(name='{self.model_name}', embedding_types={self.embedding_types})"
