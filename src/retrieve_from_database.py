from typing import List

from src.enums.database_type_enum import DatabaseType
from src.enums.transformer_library_enum import TransformerLibrary
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

def perform_search(*, vector_database_instance: VectorDatabaseInfo, search_method: str, query: str, top_k: int):
    search_type = get_search_type(search_method=search_method)

    if isinstance(search_type, DatabaseType):
        return vector_database_instance.perform_search(query=query, top_k=top_k)
    elif isinstance(search_type, TransformerLibrary):
        result = vector_database_instance.retrieve_from_database()

        # Przypisanie typów po rozpakowaniu
        text_chunks: List[str]
        chunks_metadata: List[dict]
        hash_id: List[str]
        embeddings: tuple[List, List, List] #Dense, Sparse, Colbert

        text_chunks, chunks_metadata, hash_id, embeddings = result

        print(f'embeddings len: {len(embeddings[0])} text_chunks len: {len(text_chunks)}')

        return search_type.perform_search(
            text_chunks=text_chunks,
            chunks_metadata=chunks_metadata,
            hash_id=hash_id,
            embeddings=embeddings,
            query=query,
            vector_database_instance=vector_database_instance,
            top_k=top_k
        )






