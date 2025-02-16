import gradio as gr
from src.enums.database_type import DatabaseType
from src.search_utils import count_tokens, refresh_db_list
from src.chroma_db_utils import retrieve_text_from_chroma_db
from src.lance_db_utils import retrieve_text_from_lance_db

def search_database_tab():
    with gr.Tab("🔎 Wyszukiwanie w bazie"):
        search_engine_dropdown = gr.Dropdown(
            choices=[db.value for db in DatabaseType],
            value=None,
            label="Wybierz silnik wektorowy"
        )
        db_dropdown_search = gr.Dropdown(
            choices=[],
            label="📂 Wybierz bazę (Wyszukiwanie)"
        )

        search_engine_dropdown.change(refresh_db_list, search_engine_dropdown, db_dropdown_search)

        query_input = gr.Textbox(
            label="🔎 Wpisz swoje pytanie"
        )
        top_k_slider = gr.Slider(
            1,
            100,
            10,
            step=1,
            label="🔝 Liczba najlepszych wyników"
        )
        search_btn = gr.Button("🔍 Szukaj")

        token_output = gr.Textbox(
            label="Liczba tokenów:",
            interactive=False
        )
        search_output = gr.Textbox(
            label="Wyniki wyszukiwania:",
            interactive=False
        )

        search_btn.click(ui_search_database, [search_engine_dropdown, db_dropdown_search, query_input, top_k_slider], [token_output, search_output])


def ui_search_database(db_engine: str, db_name, query, top_k):
    db_engine_enum = DatabaseType(db_engine)

    if db_engine_enum == DatabaseType.CHROMA_DB:
        retrieved_text = retrieve_text_from_chroma_db(db_name, query, top_k)
    elif db_engine_enum == DatabaseType.LANCE_DB:
        retrieved_text = retrieve_text_from_lance_db(db_name, query, top_k)

    token_count = count_tokens(retrieved_text)
    return token_count, retrieved_text
