import json

import tiktoken
from src.text_retrieval.enums.database_type_enum import DatabaseType

# Tokenizer tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Zwraca liczbę tokenów w podanym tekście, używając tiktoken.
    """
    return len(tokenizer.encode(text))


# Kiedy zmienia się "search_engine_dropdown", musimy odświeżyć listę baz
# zależnie od tego, czy to ChromaDB czy LanceDB:
def fetch_saved_databases(database_type: str):
    database_type = DatabaseType.from_display_name(database_type) # TRANSFORMACJA ENUM.VALUE NA CZYSTY ENUM
    saved_databases = database_type.db_class.get_saved_databases_from_drive_as_instances()
    list_of_list_of_labels_and_instances = []
    for key, value in saved_databases.items():
        list_of_list_of_labels_and_instances.append((f'{key} | {value.embedding_model_name} ({value.float_precision.value}) | {value.transformer_library.display_name} ({", ".join([et.value for et in value.embedding_types])}) |  Features: {", ".join(list(value.features.keys()))}  |  {value.segmentation_type.value} (Chunk Size: {value.chunk_size}, Chunk Overlap: {value.chunk_overlap}, Whole Sentences: {value.preserve_whole_sentences}, Overlap Type: {value.overlap_type.value}, Exceed limit: {value.exceed_limit}) | Number of files: {value.file_count}', json.dumps(value.to_dict())))
    return list_of_list_of_labels_and_instances


