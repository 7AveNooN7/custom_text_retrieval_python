import os
import shutil
import json
import numpy as np
from typing import List, Tuple
import gradio as gr
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel, BGEM3FlagModel
from src.config import MODEL_FOLDER
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.downloaded_model_info import DownloadedModelInfo


def get_downloaded_models_for_dropdown() -> List[Tuple[str, str]]:
    """
    Przechodzi przez podfoldery w model_folder i wyszukuje modele.
    Za model uznajemy katalog, który zawiera plik 'metadata.json'.
    Zwraca listę obiektów DownloadedModelInfo. Działa tylko przy inizjalizacji aplikacji.
    """
    downloaded_models = []

    for model_folder in os.scandir(MODEL_FOLDER):
        if model_folder.is_dir():
            metadata_json_path = os.path.join(model_folder.path, "metadata.json")
            if os.path.isfile(metadata_json_path):
                try:
                    with open(metadata_json_path, "r", encoding="utf-8") as metadata_json_file:
                        json_data = json.load(metadata_json_file)
                        model_info: DownloadedModelInfo = DownloadedModelInfo.from_dict(json_data=json_data)
                        downloaded_models.append((create_downloaded_model_label(model_info), json.dumps(model_info.to_dict(), ensure_ascii=False))) # stworze krotke (label, json_data)
                except Exception as error:
                    print(f"⚠️ Błąd odczytu metadata.json w katalogu '{model_folder.name}': {error}")

    print(f"Pobrane modele: {downloaded_models}")  # Debugging
    return downloaded_models

def create_downloaded_model_label(model_info: DownloadedModelInfo) -> str:
    model_info_dict = model_info.to_dict()

    # Lista do budowy etykiety
    label_parts = [model_info.model_name]

    # Iteracja po supported_libraries
    for library, embeddings in model_info_dict["supported_libraries"].items():
        # Tworzenie tekstu dla każdej biblioteki
        embeddings_str = ", ".join(embeddings)
        label_parts.append(f"{library}: {embeddings_str}")

    # Łączenie wszystkich części za pomocą " | "
    return "  |  ".join(label_parts)



def add_model(model_name: str) -> DownloadedModelInfo:
    safe_model_dir: str = str(model_name.replace("/", "_"))
    target_dir: str = str(os.path.join(MODEL_FOLDER, safe_model_dir))
    download_model_from_hf(model_name, target_dir)

    # Pobranie informacji o wspieranych bibliotekach
    sentence_transformers_dict = TransformerLibrary.is_sentence_transformer_model(target_dir, model_name) or {}
    flag_embedding_dict = TransformerLibrary.is_flag_embedding_model(target_dir, model_name) or {}

    # Połączenie słowników
    supported_libraries = {**sentence_transformers_dict, **flag_embedding_dict}

    # Tworzenie danych do zapisu w JSON
    metadata = {
        "model_name": model_name,
        "supported_libraries": supported_libraries
    }

    # Sprawdzenie poprawności przed zapisem
    metadata_path = os.path.join(target_dir, "metadata.json")
    try:
        json_str = json.dumps(metadata, indent=2, ensure_ascii=False)
    except TypeError as json_error:
        raise ValueError(f"Błąd przy konwersji do JSON: {json_error}")

    # Zapis do pliku metadata.json
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return DownloadedModelInfo.from_dict(json_data=metadata)




def get_targeted_model_instance(model_name: str) -> DownloadedModelInfo:
    downloaded_models = get_downloaded_models_for_dropdown()

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
