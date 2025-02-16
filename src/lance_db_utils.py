# database_lance_utils.py

import os
import json
import hashlib
import numpy as np

import lancedb
import pandas as pd
import pyarrow as pa

from src.db_utils import split_text_into_chunks, is_valid_db_name
from src.config import LANCE_DB_FOLDER
from src.embedding_model_utils import load_embedding_model


def create_new_database_lance_db(db_name: str, selected_files, chunk_size: int, chunk_overlap: int, embedding_model_name: str):
    # Walidacja nazwy bazy
    if not is_valid_db_name(db_name):
        return "❌ Niepoprawna nazwa bazy! Użyj tylko liter, cyfr, kropek i podkreśleń. Długość: 3-63 znaki."

    db_path = os.path.join(LANCE_DB_FOLDER, db_name)
    os.makedirs(db_path, exist_ok=True)

    # Load embedding model
    embedding_model = load_embedding_model(embedding_model_name)

    # Connect to LanceDB
    db = lancedb.connect(db_path)

    all_records = []

    for file_obj in selected_files:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

        for idx, chunk in enumerate(chunks):
            emb = embedding_model.encode(chunk)
            emb = np.array(emb, dtype=np.float32)  # Ensure it's a NumPy array of float32

            # Check if embedding dimension is consistent
            if emb.ndim != 1:
                return "❌ Embedding model returned an unexpected shape!"

            record_id = hashlib.md5(f"{file_obj.name}_{idx}_{chunk}".encode()).hexdigest()

            all_records.append({
                "id": record_id,
                "text": chunk,
                "source": os.path.basename(file_obj.name),
                "fragment_id": idx,
                "embedding": emb.tolist(),  # Convert NumPy array to list
                "embedding_model": embedding_model_name
            })

    if not all_records:
        return "❗ No chunks to add"

    # Determine embedding dimension
    embedding_dim = len(all_records[0]['embedding'])

    # Define schema with FixedSizeList for embedding
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("fragment_id", pa.int32()),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),  # FixedSizeList with embedding_dim
        pa.field("embedding_model", pa.string())
    ])

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Create table with defined schema
    if "data" in db.table_names():
        db.drop_table("data")
    db.create_table("data", data=df, schema=schema)

    # Save metadata
    metadata = {
        "db_name": db_name,
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    with open(os.path.join(db_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return f"✅ New LanceDB `{db_name}` has been created."


def retrieve_text_from_lance_db(db_name: str, query: str, top_k: int):
    """
    Wyszukiwanie w LanceDB bazując na kolumnie `embedding`.
    """
    db_path = os.path.join(LANCE_DB_FOLDER, db_name)
    metadata_path = os.path.join(db_path, "metadata.json")

    if not os.path.isdir(db_path):
        return "⚠️ Baza LanceDB nie istnieje!"

    model_fallback = "BAAI_bge-m3"

    # Wczytujemy info o modelu
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embedding_model_name = data.get("embedding_model", model_fallback)
    else:
        embedding_model_name = model_fallback

    embedding_model = load_embedding_model(embedding_model_name)

    # Ładujemy bazę LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table("data")  # nazwa tabeli ustalona w create_new_database_lance

    # Obliczamy embedding zapytania
    query_emb = embedding_model.encode(query).tolist()

    # W LanceDB wykorzystujemy metodę .search(col_name)
    # UWAGA: LanceDB w momencie tworzenia to wciąż projekt w rozwoju,
    #        więc syntaktyka może się różnić zależnie od wersji.
    results = (
        table.search(query_emb, "embedding")
        .limit(top_k)
        .select(["text", "source", "fragment_id", "embedding_model"])
        .to_df()
    )

    if results.empty:
        return "⚠️ Brak wyników dla podanego zapytania."

    # Wyniki – posortowane domyślnie wg podobieństwa (distance).
    # W zależności od wersji LanceDB, mogły pojawić się kolumny 'score' lub 'distance'.
    # Zakładamy, że jest kolumna 'score' (lub 'distance'), 
    # ale jeśli jest inna, trzeba dopasować:
    if "score" in results.columns:
        results = results.sort_values("score", ascending=True)
    elif "distance" in results.columns:
        results = results.sort_values("distance", ascending=True)

    # Budujemy odpowiedź
    response = ""
    for idx, row in results.iterrows():
        if "_distance" in row:
            score_val = f"distance: {row['_distance']:.4f}"
        elif "score" in row:
            score_val = f"(score: {row['score']:.4f})"
        else:
            score_val = ""
    
        response += (
            f"📄 Plik: {row['source']} "
            f"(fragment {row['fragment_id']}, {score_val}, model: {row['embedding_model']})\n"
            f"{row['text']}\n\n"
        )

    return response
