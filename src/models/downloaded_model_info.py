import json
from typing import List, Dict
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary

class DownloadedModelInfo:
    """Reprezentuje pobrany model z katalogu."""

    def __init__(self, *, model_name: str, supported_libraries: Dict[TransformerLibrary, List[EmbeddingType]]):
        self.model_name = model_name
        self.supported_libraries = supported_libraries

    @classmethod
    def from_dict(cls, *, json_data: dict):
        """Tworzy obiekt DownloadedModelInfo na podstawie danych JSON."""
        model_name = json_data.get("model_name")
        raw_supported_libraries = json_data.get("supported_libraries", {})

        if not isinstance(raw_supported_libraries, dict):
            raise ValueError("❌ 'supported_libraries' powinno być słownikiem!")

        # Konwersja kluczy na TransformerLibrary i wartości na EmbeddingType
        supported_libraries = {}
        for lib_name, embedding_list in raw_supported_libraries.items():
            try:
                # Zamiana klucza na enum TransformerLibrary
                lib_enum = TransformerLibrary.from_display_name(lib_name)

                # Zamiana wartości na enum EmbeddingType
                embedding_types = [EmbeddingType(et) for et in embedding_list]

                supported_libraries[lib_enum] = embedding_types

            except ValueError as e:
                raise ValueError(f"❌ Błąd konwersji dla '{lib_name}': {e}")

        return cls(model_name=model_name, supported_libraries=supported_libraries)

    def to_dict(self) -> dict:
        """Zwraca obiekt w formacie JSON."""
        return {
            "model_name": self.model_name,
            "supported_libraries": {
                lib.display_name: [et.value for et in embeddings]  # Użycie lib.value zamiast lib.name
                for lib, embeddings in self.supported_libraries.items()
            }
        }

    def get_supported_embeddings_from_specific_library(self, library: TransformerLibrary) -> List[EmbeddingType]:
        """Zwraca listę obsługiwanych typów embeddingów dla danej biblioteki."""
        return self.supported_libraries.get(library, [])

    def __repr__(self):
        return f"DownloadedModelInfo(name='{self.model_name}', supported_libraries={self.supported_libraries})"
