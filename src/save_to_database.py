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
from tqdm import tqdm
from transformers import AutoTokenizer

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
    step = chunk_size - chunk_overlap

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

def split_text_into_characters_chunks_by_whole_sentences(text: str, chunk_size: int, chunk_overlap: int, overlap_type: OverlapTypeEnum, exceed_limit: bool) -> List[str]:
    """
    Dzieli tekst na fragmenty z bazową długością chunk_size - chunk_overlap*2 (środkowe),
    chunk_size - chunk_overlap (pierwszy i ostatni), zachowując pełne zdania.
    Overlap "od góry" i "od dołu" jest bliski chunk_overlap, z możliwością przekroczenia o 1 zdanie.

    :param text: Tekst wejściowy.
    :param chunk_size: Orientacyjna maksymalna liczba znaków w fragmencie (wliczając spacje).
    :param chunk_overlap: Orientacyjna liczba znaków nakładania się fragmentów.
    :param overlap_type: Typ overlap
    :return: Lista fragmentów tekstu.
    """

    # Podział na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences[-1].endswith(('.', '!', '?')):
        sentences[-1] += '.'

    total_sentences_count = len(sentences)  # Całkowita liczba zdań w tekście

    final_chunks = []

    current_chunk = []
    current_length = 0
    sentence_index = 0
    if exceed_limit:
        while sentence_index < total_sentences_count:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence) + (1 if current_chunk else 0)
            # Dodajemy zdanie zawsze
            current_chunk.append(sentence)
            current_length += sentence_length
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
    elif not exceed_limit:
        while sentence_index < total_sentences_count:
            sentence = sentences[sentence_index]
            sentence_length = len(sentence) + (1 if current_chunk else 0)
            # Sprawdzamy, czy dodanie zdania nie przekroczy chunk_size
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Zamykamy bieżący chunk i dodajemy do final_chunks
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                # Tworzymy overlap, ale nie przekraczamy chunk_overlap
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_length = len(s) + (1 if overlap_sentences else 0)
                    if overlap_length + s_length > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += s_length
                # Nowy chunk zaczyna się od overlapa i dodajemy bieżące zdanie
                current_chunk = overlap_sentences
                current_length = overlap_length
                if current_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Jeśli nawet z overlapem przekracza, zaczynamy nowy chunk bez overlapa
                    current_chunk = [sentence]
                    current_length = sentence_length
            sentence_index += 1
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))
    return final_chunks

def split_text_into_tik_token_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
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
    chunks = []

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

    return chunks

def split_text_into_model_tokenizer_chunks(
        *,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
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

    chunks = []

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

        # Pobierz pozycje startu i końca fragmentu w oryginalnym tekście
        start_idx = offsets[i][0]  # Początek pierwszego tokenu we fragmencie
        end_idx = offsets[min(i + adjusted_chunk_size - 1, len(tokens) - 1)][1]  # Koniec ostatniego tokenu

        # Wyodrębnij oryginalny fragment tekstu
        chunk_text = text[start_idx:end_idx].strip()

        while len(tokenizer.encode(chunk_text, add_special_tokens=True)) > chunk_size:
            chunk_text = chunk_text[1:]

        chunks.append(chunk_text)

    return chunks

def split_text_into_tokens_chunks_by_whole_sentences(
    *,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    exceed_limit: bool,
    model_name: str,
    segmentation_type: TextSegmentationTypeEnum
) -> List[str]:

    if segmentation_type == TextSegmentationTypeEnum.TIK_TOKEN:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(DownloadedEmbeddingModel.build_target_dir(model_name))

    # Podział na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences[-1].endswith(('.', '!', '?')):
        sentences[-1] += '.'

    total_sentences_count = len(sentences)  # Całkowita liczba zdań w tekście

    final_chunks = []

    current_chunk = []
    sentence_index = 0
    if exceed_limit:
        while sentence_index < total_sentences_count:
            current_sentence = sentences[sentence_index]
            # Dodajemy zdanie zawsze
            current_chunk.append(current_sentence)
            current_sentences_string = " ".join(current_chunk)
            current_chunk_length = len(tokenizer.encode(current_sentences_string))

            # Zamykamy fragment tylko, jeśli przekroczyliśmy chunk_size i są kolejne zdania
            if current_chunk_length > chunk_size and sentence_index + 1 < total_sentences_count:
                final_chunks.append(" ".join(current_chunk))
                # Przesuwamy okno: bierzemy zdania z końca jako overlap
                overlap_sentences = []
                for current_overlap_sentence in reversed(current_chunk):
                    overlap_sentences.insert(0, current_overlap_sentence)
                    current_chunk_overlap = " ".join(overlap_sentences)
                    current_overlap_length = len(tokenizer.encode(current_chunk_overlap))
                    if current_overlap_length > chunk_overlap:
                        break
                current_chunk = overlap_sentences
            sentence_index += 1

        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    elif not exceed_limit:
        overlap_sentences_not_within_limit: List[str] = []
        sentences_not_within_limit: List[str] = []

        while sentence_index < total_sentences_count:
            # WYCIĄGAM CURRENT SENTENCE
            current_sentence = sentences[sentence_index]

            # LICZE DLUGOSC CALEGO CURRENT CHUNKA RAZEM Z CURRENT SENTENCE
            temp_current_chunk = list(current_chunk)
            temp_current_chunk.append(current_sentence)
            temp_current_chunk_string = " ".join(temp_current_chunk)
            temp_current_chunk_length = len(tokenizer.encode(temp_current_chunk_string))

            # JEZELI DLUGOSC CALEGO CURRENT CHUNKA RAZEM Z CURRENT SENTENCE JEST MNIEJSZA NIZ CHUNK_SIZE TO DODAJE DO CURRENT_CHUNK
            if temp_current_chunk_length <= chunk_size:
                current_chunk.append(current_sentence)
            else:
                # Zamykamy bieżący chunk i dodajemy do final_chunks
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                # Tworzymy overlap, ale nie przekraczamy chunk_overlap
                overlap_sentences = []
                for current_overlap_sentence in reversed(current_chunk):
                    # LICZE DLUGOSC CALEGO OVERLAP RAZEM Z CURRENT OVERLAP SENTENCE
                    temp_overlap_sentences = list(overlap_sentences)
                    temp_overlap_sentences.insert(0, current_overlap_sentence)
                    temp_overlap_sentences_string = " ".join(temp_overlap_sentences)
                    temp_overlap_length = len(tokenizer.encode(temp_overlap_sentences_string))

                    if temp_overlap_length > chunk_overlap:
                        if len(overlap_sentences) == 0:
                            overlap_sentences_not_within_limit.append(current_overlap_sentence)
                            raise ValueError(
                                f"Pojedyncze zdanie przekracza dozwolony limit overlap ({chunk_overlap} tokenów!)\n"
                                f"Zdanie ({temp_overlap_length} tokenów): {current_overlap_sentence[:60]}..."
                            )
                        break
                    overlap_sentences.insert(0, current_overlap_sentence)

                # Nowy chunk zaczyna się od overlapa i dodajemy bieżące zdanie
                current_chunk = overlap_sentences

                temp_current_chunk = list(current_chunk)
                temp_current_chunk.append(current_sentence)
                temp_current_chunk_string = " ".join(temp_current_chunk)
                temp_current_chunk_length = len(tokenizer.encode(temp_current_chunk_string))

                if temp_current_chunk_length <= chunk_size:
                    current_chunk.append(current_sentence)
                else:
                    # Jeśli nawet z overlapem przekracza, zaczynamy nowy chunk bez overlapa
                    current_chunk = [current_sentence]
                    single_sentence_len = len(tokenizer.encode(current_sentence))
                    if single_sentence_len > chunk_size:
                        raise ValueError(
                            f"Pojedyncze zdanie przekracza dozwolony limit {chunk_size} tokenów!\n"
                            f"Zdanie ({single_sentence_len} tokenów): {current_sentence[:60]}..."
                        )
            sentence_index += 1
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    return final_chunks


def generate_id(text: str, filename: str, index: int) -> str:
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()

def process_file_chunking(file_path: str, vector_database_instance: VectorDatabaseInfo) -> Tuple[List[str], List[ChunkMetadataModel]]:
    """Funkcja do przetwarzania pojedynczego pliku."""
    texts: List[str]
    metadata: List[ChunkMetadataModel]
    texts, metadata = [], []

    chunk_size: int = vector_database_instance.chunk_size
    chunk_overlap: int = vector_database_instance.chunk_overlap
    overlap_type: OverlapTypeEnum = vector_database_instance.overlap_type
    segmentation_type: TextSegmentationTypeEnum = vector_database_instance.segmentation_type
    preserve_whole_sentences: bool = vector_database_instance.preserve_whole_sentences
    exceed_limit: bool = vector_database_instance.exceed_limit
    model_name: str = vector_database_instance.embedding_model_name



    with open(file_path, "r", encoding="utf-8") as f:
        whole_text_from_file = f.read()
    if segmentation_type == TextSegmentationTypeEnum.CHARACTERS:
        if preserve_whole_sentences:
            chunks = split_text_into_characters_chunks_by_whole_sentences(
                text=whole_text_from_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                overlap_type=overlap_type,
                exceed_limit=exceed_limit
            )
        else:
            chunks = split_text_into_characters_chunks_by_characters(
                text=whole_text_from_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    elif segmentation_type == TextSegmentationTypeEnum.TIK_TOKEN or segmentation_type == TextSegmentationTypeEnum.CURRENT_MODEL_TOKENIZER:
        if preserve_whole_sentences:
            chunks = split_text_into_tokens_chunks_by_whole_sentences(
                text=whole_text_from_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                exceed_limit=exceed_limit,
                segmentation_type=segmentation_type,
                model_name=model_name
            )
        else:
            if segmentation_type == TextSegmentationTypeEnum.TIK_TOKEN:
                chunks = split_text_into_tik_token_chunks(
                    text=whole_text_from_file,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            elif vector_database_instance.segmentation_type == TextSegmentationTypeEnum.CURRENT_MODEL_TOKENIZER:
                chunks = split_text_into_model_tokenizer_chunks(
                    text=whole_text_from_file,
                    chunk_size=vector_database_instance.chunk_size,
                    chunk_overlap=vector_database_instance.chunk_overlap,
                    model_name=vector_database_instance.embedding_model_name
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

    return texts, metadata

def generate_text_chunks(vector_database_instance: VectorDatabaseInfo) -> tuple[List[str], List[ChunkMetadataModel]]:
    texts: List[str]
    metadata: List[ChunkMetadataModel]

    texts, metadata = [], []

    # Wykorzystanie ProcessPoolExecutor do wieloprocesowości
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_chunking, file_path, vector_database_instance)
                   for file_path in vector_database_instance.files_paths]

        for future in as_completed(futures):
            file_texts, file_metadata = future.result()
            texts.extend(file_texts)
            metadata.extend(file_metadata)

    return texts, metadata

def generate_embeddings(text_chunk: List[str], vector_database_instance: VectorDatabaseInfo) -> Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]:
    transformer_library: TransformerLibrary = vector_database_instance.transformer_library
    dense_embeddings, sparse_embeddings, colbert_embeddings = transformer_library.generate_embeddings(text_chunk, vector_database_instance)
    return dense_embeddings, sparse_embeddings, colbert_embeddings

def save_to_database(vector_database_instance: VectorDatabaseInfo):
    print(f'FUNCTION: save_to_database')
    text_chunks: List[str]
    chunks_metadata: List[ChunkMetadataModel]
    try:
        text_chunks, chunks_metadata = generate_text_chunks(vector_database_instance)
    except Exception as e:
        traceback.print_exc()
        gr.Warning(f'{e}')
        return

    gr.Info("✅ Text Chunks created!")
    embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]] = generate_embeddings(text_chunks, vector_database_instance)

    # try:
    #     print(f'FUNCTION: embedding_types_checking')
    #     embedding_types_checking(embeddings=embeddings, float_precision=vector_database_instance.float_precision, model_name=vector_database_instance.embedding_model_name)
    # except Exception as e:
    #     gr.Warning(f"Błąd w sprawdzaniu typów osadzeń: {e}")
    #     traceback.print_exc()  # Wyświetli pełny stack trace w terminalu
    #     return  # Zatrzymuje dalsze działanie funkcji

    gr.Info("✅ Embeddings created!")
    vector_database_instance.create_new_database(text_chunks=text_chunks, chunks_metadata=chunks_metadata, embeddings=embeddings)
    gr.Info("✅ Database created!")




