import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import gradio as gr
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)
    return chunks

def generate_id(text: str, filename: str, index: int) -> str:
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()

def process_file(file_path, vector_database_instance):
    """Funkcja do przetwarzania pojedynczego pliku."""
    texts, metadata, hash_id = [], [], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            whole_text_from_file = f.read()

        # Dzielenie tekstu na fragmenty
        chunks = split_text_into_chunks(
            whole_text_from_file,
            vector_database_instance.chunk_size,
            vector_database_instance.chunk_overlap
        )

        # Tworzenie fragmentów i metadanych
        for chunk_index, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "source": os.path.basename(file_path),
                "fragment_id": chunk_index,
            })
            hash_id.append(generate_id(chunk, file_path, chunk_index))
    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania pliku {file_path}: {e}")

    return texts, metadata, hash_id

def generate_text_chunks(vector_database_instance: VectorDatabaseInfo) -> tuple[List[str], List[dict], List[str]]:
    texts, metadata, hash_id = [], [], []

    # Wykorzystanie ProcessPoolExecutor do wieloprocesowości
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, vector_database_instance)
                   for file_path in vector_database_instance.files_paths]

        for future in as_completed(futures):
            try:
                file_texts, file_metadata, file_hash_id = future.result()
                texts.extend(file_texts)
                metadata.extend(file_metadata)
                hash_id.extend(file_hash_id)
            except Exception as e:
                print(f"❌ Błąd w procesie: {e}")

    return texts, metadata, hash_id

def generate_embeddings(text_chunk: List[str], vector_database_instance: VectorDatabaseInfo) -> Tuple[List, List, List]:
    transformer_library: TransformerLibrary = vector_database_instance.transformer_library
    dense_embeddings, sparse_embeddings, colbert_embeddings = transformer_library.generate_embeddings(text_chunk, vector_database_instance)
    return dense_embeddings, sparse_embeddings, colbert_embeddings


def save_to_database(vector_database_instance: VectorDatabaseInfo):
    text_chunks, chunks_metadata, hash_id = generate_text_chunks(vector_database_instance)
    gr.Info("✅ Text Chunks created!")
    embeddings: Tuple[List, List, List] = generate_embeddings(text_chunks, vector_database_instance)
    gr.Info("✅ Embeddings created!")
    vector_database_instance.create_new_database(text_chunks=text_chunks, chunks_metadata=chunks_metadata, hash_id=hash_id, embeddings=embeddings)
    #vector_database_instance.create_new_database_test(text_chunks=text_chunks, chunks_metadata=chunks_metadata,hash_id=hash_id, embeddings=embeddings)
    gr.Info("✅ Database created!")




