import os
import re
import json
import hashlib
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DB_FOLDER
from src.embeddings import load_embedding_model


def is_valid_db_name(name: str) -> bool:
    """
    Waliduje nazwę bazy – musi mieć 3-63 znaki i zawierać tylko [a-zA-Z0-9_-].
    """
    if not (3 <= len(name) <= 63):
        return False
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$", name):
        return False
    if ".." in name:
        return False
    return True


def generate_id(text: str, filename: str, index: int) -> str:
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")

    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


def create_new_database(db_name: str, 
                        selected_files, 
                        chunk_size: int, 
                        chunk_overlap: int, 
                        embedding_model_name: str):
    """
    Tworzy nową bazę wektorową Chroma:
      - Waliduje nazwę bazy
      - Ładuje model embeddingowy z cache
      - Przetwarza pliki na chunki
      - Dodaje dokumenty do bazy
      - Zapisuje metadane w 'metadata.json'
    """
    # Walidacja nazwy bazy
    if not is_valid_db_name(db_name):
        return "❌ Niepoprawna nazwa bazy! Użyj tylko liter, cyfr, myślników i podkreśleń. Długość: 3-63 znaki."

    # Wczytanie wybranego modelu embeddingowego (z cache)
    embedding_model = load_embedding_model(embedding_model_name)

    # Inicjalizacja (lub otwarcie) bazy wektorowej Chroma
    db_path = os.path.join(CHROMA_DB_FOLDER, db_name)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=db_name)

    texts, metadata, ids = [], [], []

    # Iterujemy po wybranych plikach
    for file_obj in selected_files:
        # file_obj.name to pełna ścieżka tymczasowa, może być różna w Gradio
        with open(file_obj.name, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        except ValueError as e:
            return f"❌ Błąd: {e}"

        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "source": os.path.basename(file_obj.name),
                "fragment_id": idx,
                "embedding_model": embedding_model_name,
            })
            ids.append(generate_id(chunk, file_obj.name, idx))

    # Dodawanie do kolekcji
    if texts:
        embeddings = embedding_model.encode(texts).tolist()
        for i in tqdm(range(len(texts)), desc="📥 Dodawanie tekstów do bazy"):
            collection.add(
                ids=[ids[i]],
                embeddings=[embeddings[i]],
                documents=[texts[i]],
                metadatas=[metadata[i]]
            )

    # Zapisujemy metadane w pliku metadata.json
    db_metadata = {
        "db_name": db_name,
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    metadata_path = os.path.join(db_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(db_metadata, f, ensure_ascii=False, indent=2)

    return f"✅ Nowa baza `{db_name}` została utworzona z użyciem modelu `{embedding_model_name}`!"


def get_databases_with_info():
    """
    Zwraca listę baz w postaci listy krotek (label, value), gdzie:
    - label: np. "nazwa_bazy | Model: X | Chunk: Y | Overlap: Z"
    - value: prawdziwa nazwa bazy (np. "nazwa_bazy")
    """
    results = []
    if not os.path.exists(CHROMA_DB_FOLDER):
        return results

    for db_name in os.listdir(CHROMA_DB_FOLDER):
        db_path = os.path.join(CHROMA_DB_FOLDER, db_name)
        if not os.path.isdir(db_path):
            continue

        model = "N/A"
        csize = "N/A"
        coverlap = "N/A"

        metadata_path = os.path.join(db_path, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("embedding_model", "N/A")
            csize = data.get("chunk_size", "N/A")
            coverlap = data.get("chunk_overlap", "N/A")

        label = f"{db_name} | Model: {model} | Chunk: {csize} | Overlap: {coverlap}"
        results.append((label, db_name))

    return results
