import os
import json
import tiktoken
import chromadb
from src.config import CHROMA_DB_FOLDER
from src.embeddings import load_embedding_model

# Tokenizer tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Zwraca liczbę tokenów w podanym tekście, używając tiktoken.
    """
    return len(tokenizer.encode(text))


def retrieve_text(db_name: str, query: str, top_k: int) -> str:
    """
    Przeszukuje wskazaną bazę (db_name) za pomocą zapytania (query),
    zwraca posortowane wyniki (do top_k).
    """
    db_path = os.path.join(CHROMA_DB_FOLDER, db_name)
    metadata_path = os.path.join(db_path, "metadata.json")

    # Domyślny fallback, gdy brak metadanych
    model_fallback = "BAAI_bge-m3"

    # Wczytujemy model z metadata.json, jeśli dostępny
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embedding_model_name = data.get("embedding_model", model_fallback)
    else:
        embedding_model_name = model_fallback

    # Ładujemy model z cache
    embedding_model = load_embedding_model(embedding_model_name)

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
