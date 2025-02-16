import os
import shutil
import json
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from src.config import MODEL_FOLDER
from src.enums.embedding_type import EmbeddingType

def list_cached_models():
    """
    Przechodzi po podfolderach w MODEL_FOLDER.
    Za model uznajemy taki folder, gdzie jest plik 'metadata.json'.
    Zwraca listę modeli w formacie (nazwa modelu z metadata.json, nazwa folderu).
    """
    models = []
    for entry in os.scandir(MODEL_FOLDER):
        if entry.is_dir():
            metadata_path = os.path.join(entry.path, "metadata.json")
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        model_name = metadata.get("model_name", entry.name)
                        models.append((model_name, entry.name))  # (label, value) dla UI
                except Exception as e:
                    print(f"⚠️ Błąd odczytu metadata.json w {entry.name}: {e}")
    return models

def load_embedding_model(model_folder_name: str):
    """
    Ładuje model Sentence Transformers z folderu:
        MODEL_FOLDER / model_folder_name
    """
    model_path = os.path.join(MODEL_FOLDER, model_folder_name)

    # Sprawdzamy, jakie embeddingi obsługuje model
    metadata_path = os.path.join(model_path, "metadata.json")
    embedding_types = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            embedding_types = metadata.get("embedding_types", [])

    print(f"✅ Ładowanie modelu z: {model_path}")
    print(f"🔹 Obsługiwane embeddingi: {embedding_types}")

    return SentenceTransformer(
        model_path,
        trust_remote_code=True
    )

def download_model_to_cache(model_name: str, selected_embedding_types: list):
    """
    Pobiera model z Hugging Face do lokalnego cache (MODEL_FOLDER) i zapisuje metadata.json.
    """
    safe_model_dir = model_name.replace("/", "_")
    target_dir = os.path.join(MODEL_FOLDER, safe_model_dir)

    # Jeśli folder już istnieje, usuwamy go
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Pobieramy model
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )

    # Tworzymy plik metadata.json w folderze modelu
    metadata = {
        "model_name": model_name,
        "embedding_types": selected_embedding_types  # Lista wybranych typów embeddingów
    }

    metadata_path = os.path.join(target_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return target_dir
