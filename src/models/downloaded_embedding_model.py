import json
import os
import shutil
from typing import List, Dict

from src.config import MODEL_FOLDER
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary

class DownloadedEmbeddingModel:
    """Reprezentuje pobrany model z katalogu."""

    def __init__(self, *, model_name: str, supported_libraries: dict[TransformerLibrary, List[EmbeddingType]]):
        self.model_name: str = model_name
        self.model_path: str = self.build_target_dir(model_name)
        self.supported_libraries: Dict[TransformerLibrary, List[EmbeddingType]] = supported_libraries

    @staticmethod
    def build_target_dir(model_name: str) -> str:
        """
        Metoda statyczna do budowania ścieżki do katalogu z modelem,
        dzięki czemu nie duplikujemy logiki w wielu miejscach.
        """
        safe_model_name = model_name.replace("/", "_")
        return os.path.join(MODEL_FOLDER, safe_model_name)

    @classmethod
    def from_huggingface(cls, model_name: str) -> "DownloadedEmbeddingModel":
        """
        Metoda klasowa do pobrania modelu z Hugging Face i utworzenia obiektu
        DownloadedEmbeddingModel. Dzięki temu nie duplikujemy logiki.
        """
        target_dir = cls.build_target_dir(model_name)
        download_model_from_hf(model_name, target_dir)

        # Tutaj wywołujemy np. metody do sprawdzenia, które biblioteki są obsługiwane
        sentence_transformers_dict: Dict[str, List[str]] = TransformerLibrary.is_sentence_transformer_model(target_dir, model_name) or {}
        flag_embedding_dict: Dict[str, List[str]] = TransformerLibrary.is_flag_embedding_model(target_dir, model_name) or {}

        supported_libraries = {
            **sentence_transformers_dict,
            **flag_embedding_dict
        }

        # 1) Tworzymy instancję na podstawie zebranych danych
        instance = cls(
            model_name=model_name,
            supported_libraries=supported_libraries
        )

        # 2) Wywołujemy to_dict(), aby stworzyć słownik z metadanymi
        metadata = instance.to_dict()

        # 3) Zapisujemy ten słownik do metadata.json
        metadata_path = os.path.join(target_dir, "metadata.json")
        try:
            json_str = json.dumps(metadata, indent=2, ensure_ascii=False)
        except TypeError as json_error:
            raise ValueError(f"Błąd przy konwersji do JSON: {json_error}")

        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        # 4) Zwracamy gotową instancję
        return instance

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



def download_model_from_hf(model_name: str, target_dir: str):
    """
    Pobiera model z Hugging Face do lokalnego cache (MODEL_FOLDER) i zapisuje metadata.json.
    """

    # Jeśli folder już istnieje, usuwamy go
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_name
    )

    # Pobieramy model
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir
    )
    return target_dir