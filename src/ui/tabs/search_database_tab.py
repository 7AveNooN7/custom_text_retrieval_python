import json
from typing import List, Tuple

import gradio as gr
from src.enums.database_type_enum import DatabaseType, DatabaseFeature
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo
from src.retrieve_from_database import perform_search, get_search_type
from src.search_utils import count_tokens, fetch_saved_databases
from src.chroma_db_utils import retrieve_text_from_chroma_db
from src.lance_db_utils import retrieve_text_from_lance_db

def search_database_tab():
    with gr.Tab("🔎 Wyszukiwanie w bazie"):

        ###################### DATABASE TYPE DROPDOWN ######################
        # STATE
        selected_database_engine_state = gr.State()
        def update_selected_database_engine_state(db_engine: str):
            return db_engine

        # COMPONENT
        @gr.render(inputs=[])
        def create_database_engine_dropdown():
            database_type_dropdown = gr.Dropdown(
                choices=[db.display_name for db in DatabaseType],
                value=None,
                label="Wybierz silnik bazy wektorowej"
            )

            # ZMIENIA STAN SAMEGO SIEBIE
            database_type_dropdown.change(
                update_selected_database_engine_state,
                database_type_dropdown,
                selected_database_engine_state
            )

            # GDY ZMIENIA SIE WYBOR SILNIKA BAZY ZERUJEMY WYBÓR Z saved_database_dropdown
            database_type_dropdown.change(
                update_selected_database_state,
                gr.State(None),
                selected_database_state
            )

            # GDY ZMIENIA SIE WYBOR SILNIKA BAZY ZERUJEMY WYBÓR Z search_method_choice
            database_type_dropdown.change(
                update_search_method_choice,
                gr.State(None),
                search_method_choice
            )


        ###################### SAVED DATABASE DROPDOWN ######################
        # STATE
        selected_database_state = gr.State()
        def update_selected_database_state(db_engine: str):
            return db_engine

        # COMPONENT
        @gr.render(inputs=[selected_database_engine_state])
        def create_saved_database_dropdown(selected_database_engine: str):
            if selected_database_engine:
                choices = fetch_saved_databases(selected_database_engine)
                saved_database_dropdown = gr.Dropdown(
                    choices=choices,
                    value=None,
                    label="📂 Wybierz bazę (Wyszukiwanie)"
                )

                saved_database_dropdown.change(
                    update_selected_database_state,
                    saved_database_dropdown,
                    selected_database_state
                )

                # # GDY ZMIENIA SIE WYBOR ZAPISANEJ BAZY ZERUJEMY WYBÓR Z search_method_choice
                saved_database_dropdown.change(
                    update_search_method_choice,
                    gr.State(None),
                    search_method_choice
                )

        ###################### QUERY TEXTBOX ######################
        query_input = gr.Textbox(
            label="🔎 Wpisz swoje pytanie"
        )

        ###################### TOP K SLIDER ######################
        top_k_slider = gr.Slider(
            1,
            100,
            10,
            step=1,
            label="🔝 Liczba najlepszych wyników"
        )

        ###################### SEARCH TYPE ######################
        # STATE
        search_method_choice = gr.State()
        def update_search_method_choice(value: str):
            return value

        # COMPONENT
        @gr.render(inputs=[selected_database_engine_state, selected_database_state])
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


        @gr.render(inputs=[search_method_choice, selected_database_engine_state, selected_database_state])
        def create_lance_db_search_options(search_method: str, database_type: str, vector_database_instance_json: str):
            if search_method and database_type and vector_database_instance_json:
                search_type = get_search_type(search_method=search_method)
                vector_database_instance = VectorDatabaseInfo.from_dict(json.loads(vector_database_instance_json))
                saved_database_supported_embeddings: List[str] = [embedding.value for embedding in
                                                                  vector_database_instance.embedding_types]
                choices = []
                if isinstance(search_type, DatabaseType):
                    database_type_enum: DatabaseType = DatabaseType.from_display_name(search_method)
                    database_supported_embeddings: List[str] = [embedding.value for embedding in database_type_enum.supported_embeddings]
                    choices = list(set(saved_database_supported_embeddings) & set(database_supported_embeddings))

                elif isinstance(search_type, TransformerLibrary):
                    library_type_enum: TransformerLibrary = TransformerLibrary.from_display_name(search_method)
                    library_supported_embeddings: List[str] = [embedding.value for embedding in library_type_enum.supported_embeddings]
                    choices = list(set(saved_database_supported_embeddings) & set(library_supported_embeddings))
                with gr.Row(
                        variant='compact',
                        equal_height=True
                ):
                    embeddings_checkboxes = gr.CheckboxGroup(
                        label='Wybierz typy wektorów, które zostaną użyte do wyszukiwania wektorowego',
                        choices=choices
                    )

                    features_choices = []
                    if search_type == DatabaseType.LANCE_DB and DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in vector_database_instance.features:
                        features_choices.append(DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value)

                    if features_choices:
                        features_checkboxes = gr.CheckboxGroup(
                            label='Inne typy wyszukiwania',
                            choices=features_choices
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
            [selected_database_engine_state, selected_database_state, query_input, top_k_slider, search_method_choice],
            [token_output, search_output]
        )



def ui_search_database(database_type: str, vector_database_instance_json: str, query: str, top_k: int, search_method: str):
    database_type_enum: DatabaseType = DatabaseType.from_display_name(database_type)
    vector_database_instance: VectorDatabaseInfo = database_type_enum.db_class.from_dict(json.loads(vector_database_instance_json))
    retrieved_text = perform_search(
        vector_database_instance=vector_database_instance,
        search_method=search_method,
        query=query,
        top_k=top_k
    )
    print(f'retrieved_text: {retrieved_text}')
    token_count = count_tokens(retrieved_text)
    return token_count, retrieved_text
