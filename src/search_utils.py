import tiktoken
import gradio as gr

from src.chroma_db_utils import get_databases_with_info_chroma_db


# Tokenizer tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Zwraca liczbę tokenów w podanym tekście, używając tiktoken.
    """
    return len(tokenizer.encode(text))


# Kiedy zmienia się "search_engine_dropdown", musimy odświeżyć listę baz
        # zależnie od tego, czy to ChromaDB czy LanceDB:
def refresh_db_list(engine_choice):
    if engine_choice == "ChromaDB":
        return gr.update(choices=get_databases_with_info_chroma_db())
    elif engine_choice == "LanceDB":
        from src.lance_db_utils import get_databases_with_info_lance_db
        return gr.update(choices=get_databases_with_info_lance_db())

