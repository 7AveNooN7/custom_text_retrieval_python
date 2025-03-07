from typing import List

from src.enums.database_type_enum import DatabaseType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.chunk_metadata_model import ChunkMetadataModel
from src.models.vector_database_info import VectorDatabaseInfo
from sentence_transformers import SentenceTransformer, util

def get_search_type(*, search_method: str):
    # Sprawdzenie w DatabaseType
    for db_type in DatabaseType:
        if search_method == db_type.display_name:  # lub db_type.name, jeśli chcesz użyć nazwy enuma
            return db_type

    # Sprawdzenie w TransformerLibrary
    for library_type in TransformerLibrary:
        if search_method == library_type.display_name:
            return library_type

    # Jeśli nie znaleziono
    return None

def perform_search(
        *,
        vector_database_instance: VectorDatabaseInfo,
        search_method: str,
        query: str,
        top_k: int,
        vector_choices: List[str],
        features_choices: List[str]
):
    search_type = get_search_type(search_method=search_method)
    response = ""
    if isinstance(search_type, DatabaseType):
        return vector_database_instance.perform_search(query=query, top_k=top_k, vector_choices=vector_choices, features_choices=features_choices)
    elif isinstance(search_type, TransformerLibrary):
        result = vector_database_instance.retrieve_from_database()

        # Przypisanie typów po rozpakowaniu
        text_chunks: List[str]
        chunks_metadata: List[ChunkMetadataModel]
        embeddings: tuple[List, List, List] #Dense, Sparse, Colbert

        text_chunks, chunks_metadata, embeddings = result

        result_text: List[str]
        result_chunks_metadata: List[ChunkMetadataModel]
        result_scores: List[float]

        result_text, result_chunks_metadata, result_scores = search_type.perform_search(
            text_chunks=text_chunks,
            chunks_metadata=chunks_metadata,
            embeddings=embeddings,
            query=query,
            vector_database_instance=vector_database_instance,
            top_k=top_k
        )

        for text, metadata, score in zip(result_text, result_chunks_metadata, result_scores):
            response += (
                f"📄 File: {metadata.source} "
                f"(fragment {metadata.fragment_id}, score: {score:.4f}, model: {vector_database_instance.embedding_model_name}, characters: {metadata.characters_count}, tokens: {metadata.tokens_count})\n"
                f"{text}\n\n"
            )

        return response







