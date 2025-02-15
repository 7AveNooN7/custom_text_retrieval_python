import os
import shutil
import re
import json
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src.config import MODEL_FOLDER


def list_cached_models():
    """
    Przechodzi po podfolderach w MODEL_FOLDER.
    Za model uznajemy taki folder, gdzie jest plik 'config.json'.
    Zwraca listę nazw folderów (bez pełnej ścieżki).
    """
    models = []
    for entry in os.scandir(MODEL_FOLDER):
        if entry.is_dir():
            config_path = os.path.join(entry.path, "config.json")
            if os.path.isfile(config_path):
                models.append(entry.name)
    return models


def load_embedding_model(model_name: str):
    """
    Ładuje model Sentence Transformers z folderu:
        MODEL_FOLDER / model_name
    """
    model_path = os.path.join(MODEL_FOLDER, model_name)
    return SentenceTransformer(
        model_path,
        #cache_folder=MODEL_FOLDER
        trust_remote_code=True
    )


def download_model_to_cache(model_name: str):
    """
    Pobiera model z Hugging Face do lokalnego cache (MODEL_FOLDER).
    """
    safe_model_dir = model_name.replace("/", "_")
    target_dir = os.path.join(MODEL_FOLDER, safe_model_dir)

    # Jeśli folder już istnieje, usuwamy go (np. przy odświeżeniu modelu)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Pobieramy snapshot z Hugging Face
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    return target_dir
