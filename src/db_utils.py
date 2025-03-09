import hashlib
import json
import os
import re
import chromadb

from src.config import CHROMA_DB_FOLDER, LANCE_DB_FOLDER, MODEL_FOLDER
from src.enums.database_type_enum import DatabaseType


def is_valid_db_name(name: str) -> bool:
    """
    Waliduje nazwę bazy – musi mieć 3-63 znaki i zawierać tylko [a-zA-Z0-9._-].
    - Dozwolone są kropki (.), ale nazwa nie może zaczynać ani kończyć się kropką.
    """
    if not (3 <= len(name) <= 63):
        return False
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$", name):
        return False
    return True





def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)
    return chunks



def get_databases_with_info(database_type: DatabaseType):
    """
    Zwraca listę baz w postaci listy krotek (label, value), gdzie:
    - label: np. "nazwa_bazy | Model: X | Chunk: Y | Overlap: Z"
    - value: prawdziwa nazwa bazy (np. "nazwa_bazy")
    """

    saved_databases = database_type.db_class.get_saved_databases_from_drive()

    print(f'saved_databases: {saved_databases}')

    database_folder = database_type.db_folder

    print(f'database_folder: {database_folder}')

    results = []
    if not os.path.exists(database_folder):
        return results

    print('CU')

    for db_folder_name in os.listdir(database_folder):
        db_path = os.path.join(database_folder, db_folder_name)
        if not os.path.isdir(db_path):
            continue

        chroma_client = chromadb.PersistentClient(path=db_path)
        # Pobieramy kolekcję
        collection = chroma_client.get_or_create_collection(name=db_folder_name)

        # Pobieramy metadane
        metadata = collection.metadata or {}
        print(f'metadata1: {metadata}')
        model = metadata.get("embedding_model", "N/A")
        csize = metadata.get("chunk_size", "N/A")
        coverlap = metadata.get("chunk_overlap", "N/A")

        label = f"{db_folder_name} | Model: {model} | Chunk: {csize} | Overlap: {coverlap}"
        results.append((label, db_folder_name))

    return results

def get_model_name_from_metadata(model_folder_name: str) -> str:
    """Pobiera rzeczywistą nazwę modelu z metadata.json w folderze modelu"""
    model_path = os.path.join(MODEL_FOLDER, model_folder_name, "metadata.json")
    if os.path.isfile(model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("model_name", model_folder_name)  # Domyślnie zwraca folder
    return model_folder_name  # Jeśli brak metadata.json