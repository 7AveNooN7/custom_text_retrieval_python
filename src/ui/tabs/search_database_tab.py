import json
from typing import List, Tuple

import gradio as gr
from src.enums.database_type_enum import DatabaseType
from src.models.vector_database_info import VectorDatabaseInfo
from src.retrieve_from_database import retrieve_from_database
from src.search_utils import count_tokens, fetch_saved_databases
from src.chroma_db_utils import retrieve_text_from_chroma_db
from src.lance_db_utils import retrieve_text_from_lance_db

def search_database_tab():
    with gr.Tab("🔎 Wyszukiwanie w bazie"):
        # WYBÓR SILNIKA BAZY DANYCH
        database_type_dropdown = gr.Dropdown(
            choices=[db.display_name for db in DatabaseType],
            value=None,
            label="Wybierz silnik bazy wektorowej"
        )


        # WYBÓR ZAPISANEJ BAZY DANYCH
        saved_database_dropdown = gr.Dropdown(
            choices=[],
            value=None,
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
        search_method_choice = gr.State()
        def update_search_method_choice(value: str):
            return value

        # GDY ZMIENIA SIE WYBOR SILNIKA BAZY ZERUJEMY WYBÓR Z search_method_choice
        database_type_dropdown.change(
            update_search_method_choice,
            gr.State(None),
            search_method_choice
        )

        # # GDY ZMIENIA SIE WYBOR ZAPISANEJ BAZY ZERUJEMY WYBÓR Z search_method_choice
        saved_database_dropdown.change(
            update_search_method_choice,
            gr.State(None),
            search_method_choice
        )

        @gr.render(inputs=[database_type_dropdown, saved_database_dropdown])
        def create_search_choices(database_type: str, vector_database_instance_json: str):
            if database_type and vector_database_instance_json:
                choices: List[Tuple[str, str]] = []
                choices.append((f'Native {database_type} search', database_type))
                vector_database_instance = VectorDatabaseInfo.from_dict(json.loads(vector_database_instance_json))
                choices.append((f'{vector_database_instance.transformer_library.display_name} search', vector_database_instance.transformer_library.display_name))
                radio_buttons = gr.Radio(
                    label='Wybierz metodę wyszukiwania',
                    choices=choices,
                    value=None
                )

                radio_buttons.change(
                    update_search_method_choice,
                    [radio_buttons],
                    [search_method_choice]
                )

        search_btn = gr.Button("🔍 Wyszukaj")

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
            [database_type_dropdown, saved_database_dropdown, query_input, top_k_slider, search_method_choice],
            []#[token_output, search_output]
        )



def ui_search_database(database_type: str, vector_database_instance_json: str, query: str, top_k: int, search_method: str):
    database_type_enum: DatabaseType = DatabaseType.from_display_name(database_type)
    vector_database_instance: VectorDatabaseInfo = database_type_enum.db_class.from_dict(json.loads(vector_database_instance_json))
    retrieve_from_database(
        vector_database_instance=vector_database_instance,
        search_method=search_method
    )
    print(f'search_method: {search_method}')
    print(f'vector_database_instance_json: {vector_database_instance_json}')
    return None
    #
    # token_count = count_tokens(retrieved_text)
    # return token_count, retrieved_text
