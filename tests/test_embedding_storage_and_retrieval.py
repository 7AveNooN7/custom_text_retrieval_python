import gc
from typing import List, Tuple, Optional

import numpy as np
import pytest
import tempfile
import shutil
import os

from src import config
from src.enums.database_type_enum import DatabaseType
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.floating_precision_enum import FloatPrecisionPointEnum
from src.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.enums.overlap_type import OverlapTypeEnum
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.chunk_metadata_model import ChunkMetadataModel
from src.save_to_database import save_to_database, generate_text_chunks, generate_embeddings


@pytest.mark.parametrize("db_type", [
    DatabaseType.CHROMA_DB,
    DatabaseType.LANCE_DB,
    DatabaseType.SQLITE,
])
def test_embedding_storage_and_retrieval(db_type):
    # Przygotuj dane testowe
    temp_dir = tempfile.mkdtemp() # temporary directory only for the test purpose
    db_path = None

    db_name = 'test_embedding_storage_and_retrieval'

    test_text = "Ala ma kota. Kot ma Ale."
    test_file_path = os.path.join(temp_dir, "test.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_text)


    try:
        # Inicjalizacja instancji bazy danych
        db_class = db_type.db_class
        vector_database_instance = db_class(
            database_name=db_name,
            embedding_model_name="BAAI/bge-m3",
            embedding_types=[EmbeddingType.DENSE],
            float_precision=FloatPrecisionPointEnum.FP16,
            segmentation_type=TextSegmentationTypeEnum.CHARACTERS,
            preserve_whole_sentences=True,
            exceed_limit=False,
            overlap_type=OverlapTypeEnum.SLIDING_WINDOW,
            chunk_size=100,
            chunk_overlap=10,
            files_paths=[test_file_path],
            transformer_library=TransformerLibrary.FlagEmbedding,
            features={}
        )

        db_folder = vector_database_instance.get_database_type().db_folder
        db_path = os.path.join(db_folder, db_name)
        if db_type == DatabaseType.SQLITE:
            db_path = db_path + '.db'

        # GENERATED VARIABLES
        generated_text_chunks: List[str]
        generated_chunks_metadata: List[ChunkMetadataModel]
        generated_text_chunks, generated_chunks_metadata = generate_text_chunks(vector_database_instance)
        generated_embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]
        generated_embeddings = generate_embeddings(generated_text_chunks, vector_database_instance)


        vector_database_instance.create_new_database(text_chunks=generated_text_chunks, chunks_metadata=generated_chunks_metadata, embeddings=generated_embeddings)

        # RETRIEVED VARIABLES
        retrieved_text_chunks: List[str]
        retrieved_chunks_metadata: List[ChunkMetadataModel]
        retrieved_embeddings: tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]
        retrieved_text_chunks, retrieved_chunks_metadata, retrieved_embeddings = vector_database_instance.retrieve_from_database()

        dense_gen, sparse_gen, colbert_gen = generated_embeddings
        dense_ret, sparse_ret, colbert_ret = retrieved_embeddings

        # TEXT CHUNKS – porównanie tekstów
        assert len(generated_text_chunks) == len(
            retrieved_text_chunks), f"TEXT CHUNKS: Mismatch in number of chunks: generated={len(generated_text_chunks)} vs retrieved={len(retrieved_text_chunks)}"

        for i in range(len(generated_text_chunks)):
            assert generated_text_chunks[i] == retrieved_text_chunks[i], (
                f"TEXT CHUNKS: Chunk {i} differs!\n"
                f"Generated: {generated_text_chunks[i]}\n"
                f"Retrieved: {retrieved_text_chunks[i]}"
            )

        assert dense_gen is not None
        assert dense_ret is not None

        # DENSE – porównanie floatów z tolerancją
        if dense_gen is not None and dense_ret is not None:
            assert len(dense_gen) == len(
                dense_ret), f"DENSE: Mismatch in number of vectors: generated={len(dense_gen)} vs retrieved={len(dense_ret)}"
            for i in range(len(dense_gen)):
                assert np.allclose(dense_gen[i], dense_ret[i], atol=1e-5), (
                    f"DENSE: Vector {i} differs!\n"
                    f"Generated: {dense_gen[i]}\n"
                    f"Retrieved: {dense_ret[i]}\n"
                    f"Diff: {np.abs(dense_gen[i] - dense_ret[i])}"
                )

        # # SPARSE – porównanie dictów
        # if sparse_gen is not None and sparse_ret is not None:
        #     assert len(sparse_gen) == len(
        #         sparse_ret), f"SPARSE: Mismatch in number of vectors: generated={len(sparse_gen)} vs retrieved={len(sparse_ret)}"
        #     for i in range(len(sparse_gen)):
        #         assert sparse_gen[i] == sparse_ret[i], (
        #             f"SPARSE: Dict {i} differs!\n"
        #             f"Generated: {sparse_gen[i]}\n"
        #             f"Retrieved: {sparse_ret[i]}"
        #         )
        #
        # # COLBERT – porównanie floatów
        # if colbert_gen is not None and colbert_ret is not None:
        #     assert len(colbert_gen) == len(
        #         colbert_ret), f"COLBERT: Mismatch in number of vectors: generated={len(colbert_gen)} vs retrieved={len(colbert_ret)}"
        #     for i in range(len(colbert_gen)):
        #         assert np.allclose(colbert_gen[i], colbert_ret[i], atol=1e-5), (
        #             f"COLBERT: Vector {i} differs!\n"
        #             f"Generated: {colbert_gen[i]}\n"
        #             f"Retrieved: {colbert_ret[i]}\n"
        #             f"Diff: {np.abs(colbert_gen[i] - colbert_ret[i])}"
        #         )

    finally:
        gc.collect()
        if os.path.isdir(db_path):
            # NIE USUWA CHROMADB BO SIE NIE DA
            if db_type != DatabaseType.CHROMA_DB:
                shutil.rmtree(db_path)
        else:
            os.remove(db_path)

