import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import gradio as gr
import numpy as np
import tiktoken
from tqdm import tqdm

from src.enums.overlap_type import OverlapTypeEnum
from src.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo

def split_text_into_chunks_by_characters(text: str, chunk_size: int, chunk_overlap: int, overlap_type: OverlapTypeEnum) -> List[str]:
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


def split_text_into_chunks_by_whole_sentences(text: str, chunk_size: int, chunk_overlap: int, overlaptype: OverlapTypeEnum) -> List[str]:
    """
    Dzieli tekst na fragmenty z bazową długością chunk_size - chunk_overlap*2 (środkowe),
    chunk_size - chunk_overlap (pierwszy i ostatni), zachowując pełne zdania.
    Overlap "od góry" i "od dołu" jest bliski chunk_overlap, z możliwością przekroczenia o 1 zdanie.

    :param text: Tekst wejściowy.
    :param chunk_size: Orientacyjna maksymalna liczba znaków w fragmencie (wliczając spacje).
    :param chunk_overlap: Orientacyjna liczba znaków nakładania się fragmentów.
    :return: Lista fragmentów tekstu.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")
    if chunk_size - chunk_overlap * 2 <= 0:
        raise ValueError("chunk_size - chunk_overlap * 2 musi być większe od 0!")

    # Podział na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences[-1].endswith(('.', '!', '?')):
        sentences[-1] += '.'

    total_sentences_count = len(sentences)  # Całkowita liczba zdań w tekście

    final_chunks = []

    if overlaptype == OverlapTypeEnum.DOUBLE:
        # Tryb 1: Double overlap "od góry i od dołu"
        # Krok 1: Tworzenie bazowych fragmentów
        base_chunks = []  # Lista, która będzie przechowywać wszystkie bazowe fragmenty
        current_chunk = []  # Aktualny fragment, do którego dodajemy zdania
        current_length = 0  # Aktualna długość current_chunk w znakach (z uwzględnieniem spacji)
        sentence_index = 0  # Indeks bieżącego zdania w liście sentences
        max_length = chunk_size - chunk_overlap if total_sentences_count > 1 else chunk_size

        while not base_chunks:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence)
            overall_length = sentence_length + (1 if current_chunk else 0)

            if current_length + overall_length > max_length and current_chunk:
                # JEŻELI DLUGOSC TEKSTU W CURRENT CHUNK PRZEKROCZY ZADANY LIMIT
                # TO ZAPISUJE LISTE STRINGOW W CURRENT_CHUNK DO BASE_CHUNKS [List[List[str]]]
                base_chunks.append(current_chunk)

                # AKUTALNE ZDANIE NIE ZOSTANIE JUZ DODANE DO CURRENT_CHUNK, WIEC TRZEBA JUZ JE DODAC PRZED NASTEPNA ITERACJA
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += overall_length

            sentence_index += 1

        max_length = chunk_size - chunk_overlap * 2

        while sentence_index < total_sentences_count:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence)
            overall_length = sentence_length + (1 if current_chunk else 0)

            if current_length + overall_length > max_length and current_chunk:
                # JEŻELI DLUGOSC TEKSTU W CURRENT CHUNK PRZEKROCZY ZADANY LIMIT
                # TO ZAPISUJE LISTE STRINGOW W CURRENT_CHUNK DO BASE_CHUNKS [List[List[str]]]
                base_chunks.append(current_chunk)

                # AKUTALNE ZDANIE NIE ZOSTANIE JUZ DODANE DO CURRENT_CHUNK, WIEC TRZEBA JUZ JE DODAC PRZED NASTEPNA ITERACJA
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += overall_length

            sentence_index += 1

        if current_chunk:
            base_chunks.append(current_chunk)

        # Krok 2: Dodawanie overlapu "od góry" i "od dołu"
        final_chunks = []
        for i, chunk in enumerate(tqdm(base_chunks, desc="Dodawanie overlapu", unit="chunk")):
            overlap_top = []
            overlap_bottom = []

            # Overlap "od góry" (zbliżony do chunk_overlap, max +1 zdanie)
            if i > 0:
                prev_chunk = base_chunks[i - 1]
                overlap_length = 0
                for sentence in reversed(prev_chunk):
                    s_length = len(sentence) + (1 if overlap_top else 0)
                    if overlap_length + s_length > chunk_overlap and overlap_length > 0:
                        break
                    overlap_top.insert(0, sentence)
                    overlap_length += s_length

            # Overlap "od dołu" (zbliżony do chunk_overlap, max +1 zdanie)
            if i < len(base_chunks) - 1:
                next_chunk = base_chunks[i + 1]
                overlap_length = 0
                for sentence in next_chunk:
                    s_length = len(sentence) + (1 if overlap_bottom else 0)
                    if overlap_length + s_length > chunk_overlap and overlap_length > 0:
                        break
                    overlap_bottom.append(sentence)
                    overlap_length += s_length

            # Łączenie: overlap_top + chunk + overlap_bottom
            final_chunk = overlap_top + chunk + overlap_bottom
            final_chunks.append(" ".join(final_chunk))
    elif overlaptype == OverlapTypeEnum.SLIDING_WINDOW:
        # Tryb 2: Sliding window z overlapem "od góry"
        final_chunks = []
        current_chunk = []
        current_length = 0
        sentence_index = 0
        while sentence_index < total_sentences_count:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence)
            overall_length = sentence_length + (1 if current_chunk else 0)
            if current_length + overall_length > chunk_size and current_chunk:
                final_chunks.append(" ".join(current_chunk))
                # Przesuwamy okno: bierzemy zdania z końca poprzedniego fragmentu jako overlap
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_length = len(s) + (1 if overlap_sentences else 0)
                    if overlap_length + s_length > chunk_overlap and overlap_length > 0:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += s_length
                current_chunk = overlap_sentences
                current_length = overlap_length
            current_chunk.append(sentence)
            current_length += overall_length
            sentence_index += 1
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    return final_chunks


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

        if vector_database_instance.segmentation_type == TextSegmentationTypeEnum.CHARACTERS:
            if vector_database_instance.preserve_whole_sentences:
                # Dzielenie tekstu na fragmenty z zachowaniem całych zdań
                chunks = split_text_into_chunks_by_whole_sentences(
                    whole_text_from_file,
                    vector_database_instance.chunk_size,
                    vector_database_instance.chunk_overlap,
                    vector_database_instance.overlap_type
                )
            else:
                chunks = split_text_into_chunks_by_characters(
                    whole_text_from_file,
                    vector_database_instance.chunk_size,
                    vector_database_instance.chunk_overlap,
                    vector_database_instance.overlap_type
                )
        else:
            chunks = split_text_into_token_chunks(
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

def generate_embeddings(text_chunk: List[str], vector_database_instance: VectorDatabaseInfo) -> Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]:
    transformer_library: TransformerLibrary = vector_database_instance.transformer_library
    dense_embeddings, sparse_embeddings, colbert_embeddings = transformer_library.generate_embeddings(text_chunk, vector_database_instance)
    return dense_embeddings, sparse_embeddings, colbert_embeddings


def save_to_database(vector_database_instance: VectorDatabaseInfo):
    text_chunks, chunks_metadata, hash_id = generate_text_chunks(vector_database_instance)
    gr.Info("✅ Text Chunks created!")
    embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]] = generate_embeddings(text_chunks, vector_database_instance)
    # print(f'text: {text_chunks[0]}')
    # print(f'dense: {embeddings[0][0]}\ntype: {type(embeddings[0][0])}')
    # print(f'sparse: {embeddings[1][0]}\ntype: {type(embeddings[1][0])}')
    # print(f'colbert: {embeddings[2][0]}\ntype: {type(embeddings[2][0])}')
    print(f'dense2: {type(embeddings[0])}')
    print(f'sparse2: {type(embeddings[1])}')
    print(f'colbert2: {type(embeddings[2])}')

    gr.Info("✅ Embeddings created!")
    vector_database_instance.create_new_database(text_chunks=text_chunks, chunks_metadata=chunks_metadata, hash_ids=hash_id, embeddings=embeddings)
    gr.Info("✅ Database created!")




