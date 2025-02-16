import os
import shutil
import json
from typing import List

from sentence_transformers import SentenceTransformer
from src.config import MODEL_FOLDER
from src.models.downloaded_model_info import DownloadedModelInfo


def get_downloaded_models() -> List[DownloadedModelInfo]:
    """
    Przechodzi przez podfoldery w model_folder i wyszukuje modele.
    Za model uznajemy katalog, który zawiera plik 'metadata.json'.
    Zwraca listę obiektów DownloadedModelInfo.
    """
    downloaded_models = []

    for model_folder in os.scandir(MODEL_FOLDER):
        if model_folder.is_dir():
            metadata_json_path = os.path.join(model_folder.path, "metadata.json")
            if os.path.isfile(metadata_json_path):
                try:
                    with open(metadata_json_path, "r", encoding="utf-8") as metadata_json_file:
                        json_data = json.load(metadata_json_file)
                        model_info = DownloadedModelInfo.from_json(json_data=json_data, folder_name=model_folder.name)
                        downloaded_models.append(model_info)
                except Exception as error:
                    print(f"⚠️ Błąd odczytu metadata.json w katalogu '{model_folder.name}': {error}")

    print(f"Pobrane modele: {downloaded_models}")  # Debugging
    return downloaded_models

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

def download_modem_from_hf(model_name: str, selected_embedding_types: list):
    """
    Pobiera model z Hugging Face do lokalnego cache (MODEL_FOLDER) i zapisuje metadata.json.
    """
    safe_model_dir = model_name.replace("/", "_")
    target_dir = os.path.join(MODEL_FOLDER, safe_model_dir)

    # Jeśli folder już istnieje, usuwamy go
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Pobieramy model
    from huggingface_hub import snapshot_download
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
