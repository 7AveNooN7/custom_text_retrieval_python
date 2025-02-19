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
                        model_info = DownloadedModelInfo.from_json(json_data=json_data)
                        downloaded_models.append(model_info)
                except Exception as error:
                    print(f"⚠️ Błąd odczytu metadata.json w katalogu '{model_folder.name}': {error}")

    print(f"Pobrane modele: {downloaded_models}")  # Debugging
    return downloaded_models


def get_targeted_model_instance(model_name: str) -> DownloadedModelInfo:
    downloaded_models = get_downloaded_models()

    # Szukamy modelu o podanej nazwie
    for model in downloaded_models:
        if model.model_name == model_name:
            return model  # Zwracamy znalezioną instancję

    raise FileNotFoundError(f"❌ Model '{model_name}' nie został znaleziony w {MODEL_FOLDER}")


def load_embedding_model(model_instance: str):
    """
    Ładuje model Sentence Transformers, szukając folderu, w którym
    wartość "model_name" w metadata.json pasuje do model_instance.name.
    """
    target_model_name = model_instance
    print(f'model_name: {target_model_name}')
    selected_model_path = None

    # Przeszukujemy MODEL_FOLDER w poszukiwaniu pasującego modelu
    for model_folder in os.scandir(MODEL_FOLDER):
        if model_folder.is_dir():
            metadata_json_path = os.path.join(model_folder.path, "metadata.json")

            if os.path.isfile(metadata_json_path):
                try:
                    with open(metadata_json_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        if metadata.get("model_name") == target_model_name:
                            selected_model_path = model_folder.path
                            break  # Znaleźliśmy pasujący model, przerywamy pętlę
                except Exception as e:
                    print(f"⚠️ Błąd podczas odczytu metadata.json w {model_folder.name}: {e}")

    if not selected_model_path:
        raise FileNotFoundError(f"❌ Nie znaleziono modelu '{target_model_name}' w katalogu {MODEL_FOLDER}")

    # Pobieramy obsługiwane embeddingi
    metadata_path = os.path.join(selected_model_path, "metadata.json")
    embedding_types = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            embedding_types = metadata.get("embedding_types", [])

    print(f"✅ Załadowano model: {target_model_name}")
    print(f"📂 Ścieżka: {selected_model_path}")
    print(f"🔹 Obsługiwane embeddingi: {embedding_types}")

    return SentenceTransformer(
        selected_model_path,
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
