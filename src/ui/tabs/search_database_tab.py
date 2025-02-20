import gradio as gr
from src.enums.database_type_enum import DatabaseType
from src.models.vector_database_info import VectorDatabaseInfo
from src.search_utils import count_tokens, fetch_saved_databases
from src.chroma_db_utils import retrieve_text_from_chroma_db
from src.lance_db_utils import retrieve_text_from_lance_db

def search_database_tab():
    with gr.Tab("🔎 Wyszukiwanie w bazie"):
        # WYBÓR SILNIKA BAZY DANYCH
        database_type_dropdown = gr.Dropdown(
            choices=[db.display_name for db in DatabaseType],
            value="Naciśnij, aby wybrać",
            allow_custom_value=True,
            label="Wybierz silnik bazy wektorowej"
        )

        # WYBÓR ZAPISANEJ BAZY DANYCH
        saved_database_dropdown = gr.Dropdown(
            choices=[],
            label="📂 Wybierz bazę (Wyszukiwanie)"
        )

        database_type_dropdown.change(
            fetch_saved_databases,
            database_type_dropdown,
            saved_database_dropdown
        )

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

        search_btn.click(
            ui_search_database,
            [saved_database_dropdown, database_type_dropdown, query_input, top_k_slider],
            []#[token_output, search_output]
        )


def ui_search_database(metadata_dict: dict, database_type_dropdown: str, query: str, top_k: int):
    print('CHUJNIA')
    database_type = DatabaseType.from_display_name(database_type_dropdown)
    #database_type.db_class.to_dict(metadata_dict)
    print(f'chuj: {metadata_dict}')
    # db_engine_enum = DatabaseType(db_engine)
    #
    if db_engine_enum == DatabaseType.CHROMA_DB:
        retrieved_text = retrieve_text_from_chroma_db(db_name, query, top_k)
    elif db_engine_enum == DatabaseType.LANCE_DB:
        retrieved_text = retrieve_text_from_lance_db(db_name, query, top_k)
    #
    # token_count = count_tokens(retrieved_text)
    # return token_count, retrieved_text
