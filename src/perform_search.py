import numpy as np
from typing import List, Optional, Tuple
from src.enums.database_type_enum import DatabaseType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.chunk_metadata_model import ChunkMetadataModel
from src.models.vector_database_info import VectorDatabaseInfo


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
        query_list: List[str],
        top_k: int,
        vector_choices: List[str],
        features_choices: List[str]
):
    print(f'FUNCTION: perform_search')
    search_type = get_search_type(search_method=search_method)
    result_text: List[str] = []
    result_chunks_metadata: List[ChunkMetadataModel] = []
    result_scores: List[float] = []

    final_results: List[Tuple[List[str], List[ChunkMetadataModel], List[float]]] = []

    if isinstance(search_type, DatabaseType):
        final_results = vector_database_instance.perform_search(query_list=query_list, top_k=top_k, vector_choices=vector_choices, features_choices=features_choices)[0]
    elif isinstance(search_type, TransformerLibrary):
        # Przypisanie typów po rozpakowaniu
        text_chunks: List[str]
        chunks_metadata: List[ChunkMetadataModel]
        embeddings: tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]

        text_chunks, chunks_metadata, embeddings = vector_database_instance.retrieve_from_database()

        # try:
        #     print(f'FUNCTION: embedding_types_checking')
        #     embedding_types_checking(embeddings=embeddings, float_precision=vector_database_instance.float_precision, model_name=vector_database_instance.embedding_model_name)
        # except Exception as e:
        #     gr.Error(f"❌ Błąd w sprawdzaniu typów osadzeń: {e}")
        #     traceback.print_exc()  # Wyświetli pełny stack trace w terminalu
        #     return  # Zatrzymuje dalsze działanie funkcji

        final_results = search_type.perform_search(
            text_chunks=text_chunks,
            chunks_metadata=chunks_metadata,
            embeddings=embeddings,
            query_list=query_list,
            vector_database_instance=vector_database_instance,
            top_k=top_k,
            vector_choices=vector_choices
        )

    final_response = []
    for index, final_result in enumerate(final_results):
        result_text, result_chunks_metadata, result_scores = final_result[index]
        response = ""
        for text, metadata, score in zip(result_text, result_chunks_metadata, result_scores):
            response += (
                f"📄 File: {metadata.source} "
                f"(fragment {metadata.fragment_id}, score: {score:.4f}, model: {vector_database_instance.embedding_model_name}, characters: {metadata.characters_count}, tokens: {metadata.tiktoken_tokens_count} (TikToken), {metadata.model_tokenizer_token_count} (Model Tokenizer))\n"
                f"{text}\n\n"
            )
        final_response.append(response)


    return final_response







