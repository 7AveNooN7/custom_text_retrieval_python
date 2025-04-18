import gc
from typing import List, Tuple, Optional
from collections import OrderedDict
import logging
import numpy as np
import pytest
import tempfile
import shutil
import os
import time

from src.text_retrieval.embeddings_type_checking import embedding_types_checking
from src.text_retrieval.enums.database_type_enum import DatabaseType
from src.text_retrieval.enums.embedding_type_enum import EmbeddingType
from src.text_retrieval.enums.floating_precision_enum import FloatPrecisionPointEnum
from src.text_retrieval.enums.overlap_type import OverlapTypeEnum
from src.text_retrieval.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.text_retrieval.enums.transformer_library_enum import TransformerLibrary
from src.text_retrieval.models.chunk_metadata_model import ChunkMetadataModel
from src.text_retrieval.save_to_database import generate_text_chunks, generate_embeddings

# Ustawienie logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def model_cache():
    cache = OrderedDict()
    yield cache


@pytest.fixture(scope="module")
def temp_database_resources():
    # Przygotowanie zasobów
    temp_dir = tempfile.mkdtemp()
    db_paths = {}
    corpus = (
        "The human body maintains a core temperature of approximately 37 degrees Celsius under normal conditions. "
        "Photosynthesis in green plants converts carbon dioxide and water into glucose using sunlight as an energy source. "
        "DNA replication occurs during the S phase of the cell cycle, ensuring each daughter cell receives an identical set of chromosomes. "
        "The greenhouse effect is caused by gases like carbon dioxide trapping heat in the Earth's atmosphere. "
        "Mitochondria are organelles responsible for producing ATP, the primary energy currency of the cell. "
        "The speed of light in a vacuum is approximately 299,792 kilometers per second, a universal constant."
    )

    test_file_path = os.path.join(temp_dir, "test.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    yield temp_dir, test_file_path, db_paths, corpus

    # Sprzątanie po wszystkich testach w module
    gc.collect()
    for db_path, db_type in db_paths.items():  # db_paths teraz przechowuje pary (ścieżka, db_type)
        if os.path.isdir(db_path):
            if db_type != DatabaseType.CHROMA_DB:  # Użyj db_type z db_paths
                shutil.rmtree(db_path, ignore_errors=True)
        elif os.path.isfile(db_path):
            time.sleep(1)
            os.remove(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)



@pytest.mark.parametrize(
    "db_type, embedding_types, float_precision_enum, transformer_library",
    [
        # FLAG EMBEDDING FP16 and FP32
        (DatabaseType.CHROMA_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP16, TransformerLibrary.FlagEmbedding),
        (DatabaseType.LANCE_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP16, TransformerLibrary.FlagEmbedding),
        (DatabaseType.SQLITE, [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT], FloatPrecisionPointEnum.FP16, TransformerLibrary.FlagEmbedding),
        (DatabaseType.CHROMA_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP32, TransformerLibrary.FlagEmbedding),
        (DatabaseType.LANCE_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP32, TransformerLibrary.FlagEmbedding),
        (DatabaseType.SQLITE, [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT], FloatPrecisionPointEnum.FP32, TransformerLibrary.FlagEmbedding),
        # SENTENCE TRANSFORMERS FP16 and FP32
        (DatabaseType.CHROMA_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP16, TransformerLibrary.SentenceTransformers),
        (DatabaseType.LANCE_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP16, TransformerLibrary.SentenceTransformers),
        (DatabaseType.SQLITE, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP16, TransformerLibrary.SentenceTransformers),
        (DatabaseType.CHROMA_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP32, TransformerLibrary.SentenceTransformers),
        (DatabaseType.LANCE_DB, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP32, TransformerLibrary.SentenceTransformers),
        (DatabaseType.SQLITE, [EmbeddingType.DENSE], FloatPrecisionPointEnum.FP32, TransformerLibrary.SentenceTransformers)
    ]
)
def test_embedding_storage_retrieval_search(db_type, embedding_types: List[EmbeddingType], float_precision_enum, transformer_library, temp_database_resources, model_cache, monkeypatch):
    temp_dir, test_file_path, db_paths, corpus = temp_database_resources
    monkeypatch.setattr("src.text_retrieval.enums.transformer_library_enum._model_cache", model_cache)

    db_name = f'test_{len(embedding_types)}_{float_precision_enum.value}_{transformer_library.display_name}'
    model_name = 'BAAI/bge-m3'

    try:
        # Inicjalizacja instancji bazy danych
        db_class = db_type.db_class
        vector_database_instance = db_class(
            database_name=db_name,
            embedding_model_name=model_name,
            embedding_types=embedding_types,
            float_precision=float_precision_enum,
            segmentation_type=TextSegmentationTypeEnum.CHARACTERS,
            preserve_whole_sentences=True,
            exceed_limit=False,
            overlap_type=OverlapTypeEnum.SLIDING_WINDOW,
            chunk_size=10,
            chunk_overlap=0,
            files_paths=[test_file_path],
            transformer_library=transformer_library,
            features={}
        )


        db_folder = vector_database_instance.get_database_type().db_folder
        db_path = os.path.join(db_folder, db_name)
        if db_type == DatabaseType.SQLITE:
            db_path = db_path + '.db'
        if os.path.exists(db_path):
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
            else:
                os.remove(db_path)

        db_paths[db_path] = db_type

        start_emb = time.time()
        # GENERATED VARIABLES
        generated_text_chunks: List[str]
        generated_chunks_metadata: List[ChunkMetadataModel]
        generated_text_chunks, generated_chunks_metadata = generate_text_chunks(vector_database_instance)
        generated_embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]
        generated_embeddings = generate_embeddings(generated_text_chunks, vector_database_instance)
        print(f"Czas generowania embeddingów: {time.time() - start_emb:.2f} s")


        vector_database_instance.create_new_database(text_chunks=generated_text_chunks, chunks_metadata=generated_chunks_metadata, embeddings=generated_embeddings)

        # RETRIEVED VARIABLES
        retrieved_text_chunks: List[str]
        retrieved_chunks_metadata: List[ChunkMetadataModel]
        retrieved_embeddings: tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]
        retrieved_text_chunks, retrieved_chunks_metadata, retrieved_embeddings = vector_database_instance.retrieve_from_database()

        dense_gen, sparse_gen, colbert_gen = generated_embeddings
        dense_ret, sparse_ret, colbert_ret = retrieved_embeddings


        embedding_types_checking(
            embeddings=generated_embeddings,
            float_precision=float_precision_enum,
            model_name=model_name
        )

        embedding_types_checking(
            embeddings=retrieved_embeddings,
            float_precision=float_precision_enum,
            model_name=model_name
        )

        # TEXT CHUNKS – porównanie tekstów
        assert len(generated_text_chunks) == len(
            retrieved_text_chunks), f"TEXT CHUNKS: Mismatch in number of chunks: generated={len(generated_text_chunks)} vs retrieved={len(retrieved_text_chunks)}"

        for generated_chunk, retrieved_chunk in zip(generated_text_chunks, retrieved_text_chunks):
            assert generated_chunk == retrieved_chunk


        for i in range(len(generated_text_chunks)):
            assert generated_text_chunks[i] == retrieved_text_chunks[i], (
                f"TEXT CHUNKS: Chunk {i} differs!\n"
                f"Generated: {generated_text_chunks[i]}\n"
                f"Retrieved: {retrieved_text_chunks[i]}"
            )
            assert generated_chunks_metadata[i].to_dict() == retrieved_chunks_metadata[i].to_dict(), f"METADATA CHUNKS: Chunk {i} differs!"

        assert dense_gen is not None
        assert dense_ret is not None

        if db_type == TransformerLibrary.FlagEmbedding:
            assert sparse_gen is not None
            assert colbert_gen is not None

        # DENSE – porównanie floatów z tolerancją
        if dense_gen is not None and dense_ret is not None:
            assert len(dense_gen) == len(
                dense_ret), f"DENSE: Mismatch in number of vectors: generated={len(dense_gen)} vs retrieved={len(dense_ret)}"
            for i in range(len(dense_gen)):
                assert np.array_equal(dense_gen[i], dense_ret[i]), (
                    f"DENSE: Vector {i} differs!\n"
                    f"Generated: {dense_gen[i]}\n"
                    f"Retrieved: {dense_ret[i]}\n"
                    f"Diff: {dense_gen[i] - dense_ret[i]}"
                )

        # SPARSE – porównanie dictów
        if sparse_gen is not None and sparse_ret is not None:
            assert len(sparse_gen) == len(
                sparse_ret), f"SPARSE: Mismatch in number of vectors: generated={len(sparse_gen)} vs retrieved={len(sparse_ret)}"
            for i in range(len(sparse_gen)):
                assert sparse_gen[i] == sparse_ret[i], (
                    f"SPARSE: Dict {i} differs!\n"
                    f"Generated: {sparse_gen[i]}\n"
                    f"Retrieved: {sparse_ret[i]}"
                )

        # COLBERT – porównanie floatów
        if colbert_gen is not None and colbert_ret is not None:
            assert len(colbert_gen) == len(
                colbert_ret), f"COLBERT: Mismatch in number of vectors: generated={len(colbert_gen)} vs retrieved={len(colbert_ret)}"
            for i in range(len(colbert_gen)):
                assert np.allclose(colbert_gen[i], colbert_ret[i]), (
                    f"COLBERT: Vector {i} differs!\n"
                    f"Generated: {colbert_gen[i]}\n"
                    f"Retrieved: {colbert_ret[i]}\n"
                    f"Diff: {np.abs(colbert_gen[i] - colbert_ret[i])}"
                )

        queries = [
            "How does the body regulate its internal heat?",
            "What is the end product of the process that plants use to harness solar energy?",
            "At what stage do cells duplicate their genetic material?",
            "Why does the Earth's climate retain warmth?",
            "Which cellular structure generates the energy molecule used by cells?",
            "What is the velocity of electromagnetic waves in empty space?",
        ]

        if db_type != DatabaseType.SQLITE:
            # final_results: List[Tuple[List[str], List[ChunkMetadataModel], List[float]]] = []
            final_results = vector_database_instance.perform_search(query_list=queries, top_k=3, vector_choices=[et.value for et in embedding_types], features_choices={})

            for index, query_result in enumerate(final_results):
                text_results = query_result[0] # [0] wybiera z tuple text: Tuple[List[str]...
                top_text_result = text_results[0]  # [0] bo sprawdzam najwiekszy score: List[str]
                assert top_text_result == generated_text_chunks[index]
                assert top_text_result == retrieved_text_chunks[index]
                for text_result in text_results:
                    assert text_result in generated_text_chunks
                    assert text_result in retrieved_text_chunks

        final_results = transformer_library.perform_search(
            text_chunks=retrieved_text_chunks,
            chunks_metadata=retrieved_chunks_metadata,
            embeddings=retrieved_embeddings,
            query_list=queries,
            vector_database_instance=vector_database_instance,
            top_k=3,
            vector_choices=[et.value for et in embedding_types]
        )

        for index, query_result in enumerate(final_results):
            text_results = query_result[0]  # [0] wybiera z tuple text: Tuple[List[str]...
            top_text_result = text_results[0]  # [0] bo sprawdzam najwiekszy score: List[str]
            assert top_text_result == generated_text_chunks[index]
            assert top_text_result == retrieved_text_chunks[index]
            for text_result in text_results:
                assert text_result in generated_text_chunks
                assert text_result in retrieved_text_chunks

    finally:
        gc.collect()

