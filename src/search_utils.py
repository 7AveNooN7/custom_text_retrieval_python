import tiktoken
import gradio as gr
from markdown_it.cli.parse import interactive

from src.db_utils import get_databases_with_info
from src.enums.database_type_enum import DatabaseType

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
    print(f'database_type_1: {database_type}')
    database_type = DatabaseType.from_display_name(database_type) # TRANSFORMACJA ENUM.VALUE NA CZYSTY ENUM
    saved_databases = database_type.db_class.get_saved_databases_from_drive_as_instances()
    list_of_list_of_labels_and_instances = []
    for key, value in saved_databases.items():
        list_of_list_of_labels_and_instances.append((f'{key} | Model: {value.embedding_model_name} | Embeddings: {[et.value for et in value.embedding_types]} | Chunk Size: {value.chunk_size} | Chunk Overlap: {value.chunk_overlap} | Number of files: {value.file_count}', value.to_dict()))
        print(f'value: {value.to_dict()}')
    return gr.update(choices=list_of_list_of_labels_and_instances)


