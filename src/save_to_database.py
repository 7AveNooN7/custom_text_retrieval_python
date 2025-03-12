import hashlib
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import gradio as gr
import numpy as np
import tiktoken
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer

from src.embeddings_type_checking import embedding_types_checking
from src.enums.overlap_type import OverlapTypeEnum
from src.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.chunk_metadata_model import ChunkMetadataModel
from src.models.downloaded_embedding_model import DownloadedEmbeddingModel
from src.models.vector_database_info import VectorDatabaseInfo


def split_text_into_characters_chunks_by_characters(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Dzieli tekst na fragmenty o określonej liczbie znaków z paskiem postępu.

    :param text: Tekst wejściowy.
    :param chunk_size: Maksymalna liczba znaków w fragmencie.
    :param chunk_overlap: Nakładanie się fragmentów w znakach.
    :return: Lista fragmentów tekstu.
    """
    # Walidacja parametrów
    if chunk_size <= 0:
        raise ValueError("Parametr 'chunk_size' musi być większy od 0!")
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")

    # Jeśli tekst jest krótszy niż chunk_size, zwróć go jako jeden fragment
    if len(text) <= chunk_size:
        return [text]

    chunks = []

    # Główna pętla dzielenia tekstu
    for i in tqdm(range(0, len(text), step), desc="Characters chunks: Przetwarzanie fragmentów",
                  unit="chunk", total=math.ceil(len(text) / step)):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

def split_text_into_characters_chunks_by_whole_sentences(text: str, chunk_size: int, chunk_overlap: int, overlaptype: OverlapTypeEnum) -> List[str]:
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

    # if overlaptype == OverlapTypeEnum.DOUBLE:
    #     # Tryb 1: Double overlap "od góry i od dołu"
    #     # Krok 1: Tworzenie bazowych fragmentów
    #     base_chunks = []  # Lista, która będzie przechowywać wszystkie bazowe fragmenty
    #     current_chunk = []  # Aktualny fragment, do którego dodajemy zdania
    #     current_length = 0  # Aktualna długość current_chunk w znakach (z uwzględnieniem spacji)
    #     sentence_index = 0  # Indeks bieżącego zdania w liście sentences
    #
    #     # Pierwsza pętla: pierwszy fragment
    #     max_length = chunk_size - chunk_overlap if total_sentences_count > 1 else chunk_size
    #     while not base_chunks and sentence_index < total_sentences_count:
    #         sentence = sentences[sentence_index]
    #         sentence_length = len(sentence)
    #         overall_length = sentence_length + (1 if current_chunk else 0)
    #
    #         # Dodajemy zdanie zawsze, nawet jeśli przekracza max_length
    #         current_chunk.append(sentence)
    #         current_length += overall_length
    #
    #         # Jeśli przekroczyliśmy max_length i mamy więcej zdań, zamykamy fragment
    #         if current_length > max_length and sentence_index + 1 < total_sentences_count:
    #             base_chunks.append(current_chunk)
    #             current_chunk = []
    #             current_length = 0
    #
    #         sentence_index += 1
    #
    #     if current_chunk and not base_chunks:
    #         base_chunks.append(current_chunk)
    #         current_chunk = []
    #         current_length = 0
    #
    #     # Druga pętla: środkowe i ostatni fragment
    #     max_length = chunk_size - chunk_overlap * 2
    #     while sentence_index < total_sentences_count:
    #         sentence = sentences[sentence_index]
    #         sentence_length = len(sentence)
    #         overall_length = sentence_length + (1 if current_chunk else 0)
    #         # Dodajemy zdanie zawsze
    #         current_chunk.append(sentence)
    #         current_length += overall_length
    #         # Zamykamy fragment, jeśli przekroczyliśmy max_length i są kolejne zdania
    #         if current_length > max_length and sentence_index + 1 < total_sentences_count:
    #             base_chunks.append(current_chunk)
    #             current_chunk = []
    #             current_length = 0
    #         sentence_index += 1
    #
    #     if current_chunk:
    #         base_chunks.append(current_chunk)
    #
    #     for i, chunk in enumerate(base_chunks):
    #         chunk_text = " ".join(chunk)
    #
    #         print(f'{i} ({len(chunk_text)}): {chunk_text}')
    #
    #     # Krok 2: Dodawanie overlapu "od góry" i "od dołu"
    #     final_chunks = []
    #     for i, chunk in enumerate(tqdm(base_chunks, desc="Dodawanie overlapu", unit="chunk")):
    #         overlap_top = []
    #         overlap_bottom = []
    #
    #         # Overlap "od góry"
    #         if i > 0:
    #             prev_chunk = base_chunks[i - 1]
    #             overlap_length = 0
    #             for sentence in reversed(prev_chunk):
    #                 sentence_length = len(sentence) + (1 if overlap_top else 0)
    #                 overlap_top.insert(0, sentence)
    #                 overlap_length += sentence_length
    #                 if overlap_length > chunk_overlap:
    #                     #print(f'overlap_length: {overlap_length}, fragment: {i}')
    #                     break
    #         # Overlap "od dołu"
    #         if i < len(base_chunks) - 1:
    #             next_chunk = base_chunks[i + 1]
    #             overlap_length = 0
    #             for sentence in next_chunk:
    #                 sentence_length = len(sentence) + (1 if overlap_bottom else 0)
    #                 overlap_bottom.append(sentence)
    #                 overlap_length += sentence_length
    #                 if overlap_length > chunk_overlap:
    #                     break
    #         # Łączenie: overlap_top + chunk + overlap_bottom
    #         final_chunk = overlap_top + chunk + overlap_bottom
    #         final_chunks.append(" ".join(final_chunk))

    if overlaptype == OverlapTypeEnum.SLIDING_WINDOW:
        # Tryb 2: Sliding window z overlapem "od góry"
        current_chunk = []
        current_length = 0
        sentence_index = 0

        while sentence_index < total_sentences_count:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence)
            overall_length = sentence_length + (1 if current_chunk else 0)

            # Dodajemy zdanie zawsze
            current_chunk.append(sentence)
            current_length += overall_length

            # Zamykamy fragment tylko, jeśli przekroczyliśmy chunk_size i są kolejne zdania
            if current_length > chunk_size and sentence_index + 1 < total_sentences_count:
                final_chunks.append(" ".join(current_chunk))
                # Przesuwamy okno: bierzemy zdania z końca jako overlap
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_length = len(s) + (1 if overlap_sentences else 0)
                    overlap_sentences.insert(0, s)
                    overlap_length += s_length
                    if overlap_length > chunk_overlap:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length

            sentence_index += 1

        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    # for i, chunk in enumerate(final_chunks):
    #     print(f'{i}: {chunk}')

    return final_chunks


def split_text_into_token_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    text_segmentation_type: TextSegmentationTypeEnum,
    model_name: str
) -> List[str]:
    """
    Dzieli tekst na fragmenty o określonej liczbie tokenów z paskiem postępu.

    :param text: Tekst wejściowy.
    :param chunk_size: Maksymalna liczba tokenów w każdym fragmencie.
    :param chunk_overlap: Liczba tokenów nakładania się między fragmentami.
    :param text_segmentation_type: Typ segmentacji tekstu.
    :param model_name: Nazwa modelu dla tokenizera.
    :return: Lista fragmentów tekstu.
    """
    # Walidacja parametrów
    if chunk_size <= 0:
        raise ValueError("Parametr 'chunk_size' musi być większy od 0!")

    chunks = []

    # Wybór tokenizera
    if text_segmentation_type == TextSegmentationTypeEnum.TIK_TOKEN:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)

        if not tokens:
            return [""]
        if len(tokens) <= chunk_size:
            return [text]

        step = chunk_size - chunk_overlap

        # Główna pętla dzielenia tokenów
        for i in tqdm(range(0, len(tokens), step), desc="Token chunks: Przetwarzanie fragmentów",
                      unit="chunk", total=math.ceil(len(tokens) / step)):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

    elif text_segmentation_type == TextSegmentationTypeEnum.CURRENT_MODEL_TOKENIZER:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens_count = len(tokenizer.encode("", add_special_tokens=True))
        adjusted_chunk_size = chunk_size - special_tokens_count
        if adjusted_chunk_size <= 0:
            raise ValueError("chunk_size jest za mały po uwzględnieniu tokenów specjalnych!")
        # Tokenizacja z zachowaniem informacji o pozycjach
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.tokens()
        offsets = encoding.offset_mapping  # Mapowanie pozycji znaków w oryginalnym tekście

        if not tokens:
            return [""]
        if len(tokens) <= adjusted_chunk_size:
            return [text]

        step = adjusted_chunk_size - chunk_overlap

        for i in tqdm(range(0, len(tokens), step), desc="Token chunks: Przetwarzanie fragmentów",
                      unit="chunk", total=math.ceil(len(tokens) / step)):
            chunk_tokens = tokens[i:i + adjusted_chunk_size]

            # Pobierz pozycje startu i końca fragmentu w oryginalnym tekście
            start_idx = offsets[i][0]  # Początek pierwszego tokenu w fragmencie
            end_idx = offsets[min(i + adjusted_chunk_size - 1, len(tokens) - 1)][1]  # Koniec ostatniego tokenu

            # Wyodrębnij oryginalny fragment tekstu
            chunk_text = text[start_idx:end_idx].strip()

            while len(tokenizer.encode(chunk_text, add_special_tokens=True)) > chunk_size:
                chunk_text = chunk_text[1:]

            chunks.append(chunk_text)

    else:
        raise Exception("Nieprawidłowy typ segmentacji!")

    return chunks


def generate_id(text: str, filename: str, index: int) -> str:
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()

def process_file(file_path: str, vector_database_instance: VectorDatabaseInfo) -> Tuple[List[str], List[ChunkMetadataModel]]:
    """Funkcja do przetwarzania pojedynczego pliku."""
    texts: List[str]
    metadata: List[ChunkMetadataModel]
    texts, metadata = [], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            whole_text_from_file = f.read()

        if vector_database_instance.segmentation_type == TextSegmentationTypeEnum.CHARACTERS:
            if vector_database_instance.preserve_whole_sentences:
                # Dzielenie tekstu na fragmenty z zachowaniem całych zdań
                chunks = split_text_into_characters_chunks_by_whole_sentences(
                    whole_text_from_file,
                    vector_database_instance.chunk_size,
                    vector_database_instance.chunk_overlap,
                    vector_database_instance.overlap_type
                )
            else:
                chunks = split_text_into_characters_chunks_by_characters(
                    whole_text_from_file,
                    vector_database_instance.chunk_size,
                    vector_database_instance.chunk_overlap
                )
        else:
            chunks = split_text_into_token_chunks(
                whole_text_from_file,
                vector_database_instance.chunk_size,
                vector_database_instance.chunk_overlap,
                vector_database_instance.segmentation_type,
                vector_database_instance.embedding_model_name
            )

        tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")

        model_tokenizer = AutoTokenizer.from_pretrained(DownloadedEmbeddingModel.build_target_dir(vector_database_instance.embedding_model_name))

        # Tworzenie fragmentów i metadanych
        for chunk_index, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append(
                ChunkMetadataModel(
                    source=os.path.basename(file_path),
                    hash_id=generate_id(chunk, file_path, chunk_index),
                    fragment_id=chunk_index,
                    characters_count=len(chunk),
                    tiktoken_tokens_count=len(tiktoken_tokenizer.encode(chunk)),
                    model_tokenizer_token_count=len(model_tokenizer.encode(chunk))
                )
            )
    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania pliku {file_path}: {e}")

    return texts, metadata

def generate_text_chunks(vector_database_instance: VectorDatabaseInfo) -> tuple[List[str], List[ChunkMetadataModel]]:
    texts: List[str]
    metadata: List[ChunkMetadataModel]

    texts, metadata = [], []

    # Wykorzystanie ProcessPoolExecutor do wieloprocesowości
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, vector_database_instance)
                   for file_path in vector_database_instance.files_paths]

        for future in as_completed(futures):
            try:
                file_texts, file_metadata = future.result()
                texts.extend(file_texts)
                metadata.extend(file_metadata)
            except Exception as e:
                print(f"❌ Błąd w procesie: {e}")

    return texts, metadata

def generate_embeddings(text_chunk: List[str], vector_database_instance: VectorDatabaseInfo) -> Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]:
    transformer_library: TransformerLibrary = vector_database_instance.transformer_library
    dense_embeddings, sparse_embeddings, colbert_embeddings = transformer_library.generate_embeddings(text_chunk, vector_database_instance)
    return dense_embeddings, sparse_embeddings, colbert_embeddings

def save_to_database(vector_database_instance: VectorDatabaseInfo):
    text_chunks: List[str]
    chunks_metadata: List[ChunkMetadataModel]
    text_chunks, chunks_metadata = generate_text_chunks(vector_database_instance)
    gr.Info("✅ Text Chunks created!")
    embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]] = generate_embeddings(text_chunks, vector_database_instance)

    try:
        embedding_types_checking(embeddings)
    except Exception as e:
        gr.Error(f"❌ Błąd w sprawdzaniu typów osadzeń: {e}")
        traceback.print_exc()  # Wyświetli pełny stack trace w terminalu
        return  # Zatrzymuje dalsze działanie funkcji

    gr.Info("✅ Embeddings created!")
    vector_database_instance.create_new_database(text_chunks=text_chunks, chunks_metadata=chunks_metadata, embeddings=embeddings)
    gr.Info("✅ Database created!")




