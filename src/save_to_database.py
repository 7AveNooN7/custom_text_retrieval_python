import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import gradio as gr
import tiktoken
from tqdm import tqdm

from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Dzieli tekst na fragmenty o określonej liczbie znaków z paskiem postępu.

    :param text: Tekst wejściowy.
    :param chunk_size: Maksymalna liczba znaków w fragmencie.
    :param chunk_overlap: Nakładanie się fragmentów w znakach.
    :return: Lista fragmentów tekstu.
    """
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")

    chunks = []
    total_iterations = (len(text) - chunk_overlap) // step + 1  # Liczba iteracji pętli

    for i in tqdm(range(0, len(text), step), desc="Characters chunks: Przetwarzanie fragmentów", unit="chunk", total=total_iterations):
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)

    return chunks


def split_text_into_chunks1(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Dzieli tekst na fragmenty zawierające pełne zdania, wykorzystując regex do detekcji końców zdań.

    :param text: Tekst wejściowy.
    :param chunk_size: Maksymalna liczba znaków w fragmencie.
    :param chunk_overlap: Nakładanie się fragmentów w znakach.
    :return: Lista fragmentów tekstu.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")

    # Podział na zdania za pomocą regexu (obsługa . ! ? jako końce zdań)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0
    overlap_sentences = []
    overlap_length = 0  # Dokładna długość overlapu

    for sentence in tqdm(sentences, desc="Tworzenie chunków", unit="chunk"):
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size:
            # Dodanie dokładnego overlapu (sumujemy zdania do chunk_overlap znaków)
            if chunk_overlap > 0:
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):  # Przechodzimy od końca chunka
                    if overlap_length + len(s) > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)  # Dodajemy na początek (kolejność ma znaczenie)
                    overlap_length += len(s)

            # Zapisujemy aktualny chunk
            chunks.append(" ".join(current_chunk))

            # Resetujemy nowy chunk, startując od overlapu
            current_chunk = overlap_sentences.copy()
            current_length = overlap_length

        # Dodajemy zdanie do aktualnego chunka
        current_chunk.append(sentence)
        current_length += sentence_length

    # Dodanie ostatniego chunka
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_text_into_token_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Dzieli tekst na fragmenty o określonej liczbie tokenów z paskiem postępu.

    :param text: Tekst wejściowy.
    :param chunk_size: Maksymalna liczba tokenów w fragmencie.
    :param chunk_overlap: Nakładanie się fragmentów w tokenach.
    :return: Lista fragmentów tekstu.
    """
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0

    # Oblicz liczbę iteracji, aby pasek postępu działał poprawnie
    total_iterations = (len(tokens) - chunk_overlap) // (chunk_size - chunk_overlap) + 1

    for _ in tqdm(range(total_iterations), desc="Token chunks: Trwa przetwarzanie fragmentów", unit="chunk"):
        if start >= len(tokens):
            break
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += chunk_size - chunk_overlap  # Przesunięcie o chunk_size - chunk_overlap

    return chunks

def generate_id(text: str, filename: str, index: int) -> str:
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()

def process_file(file_path: str, vector_database_instance: VectorDatabaseInfo):
    """Funkcja do przetwarzania pojedynczego pliku."""
    texts, metadata, hash_id = [], [], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            whole_text_from_file = f.read()

        # Dzielenie tekstu na fragmenty
        chunks = split_text_into_chunks1(
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
    #print(f'text: {text_chunks[0]}')
    # print(f'dense: {embeddings[0][0]}')
    # #print(f'sparse: {embeddings[1][0]}')
    # print(f'colbert: {embeddings[2][0]}')
    gr.Info("✅ Embeddings created!")
    vector_database_instance.create_new_database(text_chunks=text_chunks, chunks_metadata=chunks_metadata, hash_id=hash_id, embeddings=embeddings)
    gr.Info("✅ Database created!")




