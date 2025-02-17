import os
import json
import chromadb
from tqdm import tqdm
from src.db_utils import generate_id, is_valid_db_name, split_text_into_chunks
from src.config import CHROMA_DB_FOLDER
from src.embedding_model_utils import load_embedding_model, get_targeted_model_instance
from src.models.downloaded_model_info import DownloadedModelInfo

def create_new_database_chroma_db(db_name: str,
                                  selected_files,
                                  chunk_size: int,
                                  chunk_overlap: int,
                                  model_instance: DownloadedModelInfo):
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
        return "❌ Niepoprawna nazwa bazy! Użyj tylko liter, cyfr, kropek i podkreśleń. Długość: 3-63 znaki."

    # Wczytanie wybranego modelu embeddingowego (z cache)
    print(f'create_new_database_chroma_db model_instance:name: {model_instance.model_name}')
    embedding_model = load_embedding_model(model_instance)

    # Inicjalizacja (lub otwarcie) bazy wektorowej Chroma
    db_path = os.path.join(CHROMA_DB_FOLDER, db_name)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=db_name
    )

    texts, metadata, ids = [], [], []

    # Iterujemy po wybranych plikach
    for file_obj in selected_files:
        # file_obj.name to pełna ścieżka tymczasowa, może być różna w Gradio
        with open(file_obj.model_name, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        except ValueError as e:
            return f"❌ Błąd: {e}"

        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "source": os.path.basename(file_obj.model_name),
                "fragment_id": idx,
                "embedding_model": model_instance.model_name,
            })
            ids.append(generate_id(chunk, file_obj.model_name, idx))

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
        "embedding_model": model_instance.model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    metadata_path = os.path.join(db_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(db_metadata, f, ensure_ascii=False, indent=2)

    return f"✅ Nowa baza `{db_name}` została utworzona z użyciem modelu `{model_instance.model_name}`!"


def retrieve_text_from_chroma_db(db_name: str, query: str, top_k: int) -> str:
    """
    Przeszukuje wskazaną bazę (db_name) za pomocą zapytania (query),
    zwraca posortowane wyniki (do top_k).
    """
    db_path = os.path.join(CHROMA_DB_FOLDER, db_name)
    metadata_path = os.path.join(db_path, "metadata.json")


    # Wczytujemy model z metadata.json, jeśli dostępny
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embedding_model_name = data.get("embedding_model")

    model_instance = get_targeted_model_instance(embedding_model_name)

    # Ładujemy model z cache
    embedding_model = load_embedding_model(model_instance)

    # Inicjalizacja bazy Chroma
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=db_name)

    # Embedding zapytania
    query_embedding = embedding_model.encode([query]).tolist()

    # Zapytanie do bazy
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    if not results["documents"]:
        return "⚠️ Brak wyników dla podanego zapytania."

    # Rozpakowujemy i sortujemy po dystansie (im mniejszy, tym bliżej)
    sorted_results = sorted(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        key=lambda x: x[2]
    )

    # Budujemy odpowiedź tekstową
    response = ""
    for doc, meta, dist in sorted_results:
        response += (
            f"📄 Plik: {meta['source']} "
            f"(fragment {meta['fragment_id']}, dystans: {dist:.4f}, model: {meta.get('embedding_model')})\n"
            f"{doc}\n\n"
        )

    return response
