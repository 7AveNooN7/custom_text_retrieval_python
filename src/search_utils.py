import tiktoken
import gradio as gr
from src.db_utils import get_databases_with_info
from src.enums.database_type import DatabaseType

# Tokenizer tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Zwraca liczbę tokenów w podanym tekście, używając tiktoken.
    """
    return len(tokenizer.encode(text))


# Kiedy zmienia się "search_engine_dropdown", musimy odświeżyć listę baz
# zależnie od tego, czy to ChromaDB czy LanceDB:
def refresh_db_list(database_type: str):
    database_type = DatabaseType(database_type) # TRANSFORMACJA ENUM.VALUE NA CZYSTY ENUM
    return gr.update(choices=get_databases_with_info(database_type))


