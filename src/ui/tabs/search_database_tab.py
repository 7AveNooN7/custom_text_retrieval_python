import json
from typing import List, Tuple

import gradio as gr
from src.enums.database_type_enum import DatabaseType, DatabaseFeature
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo
from src.perform_search import perform_search, get_search_type
from src.search_utils import count_tokens, fetch_saved_databases

def search_database_tab():
    with ((gr.Tab("🔎 Wyszukiwanie w bazie"))):

        ###################### DATABASE TYPE DROPDOWN ######################
        # STATE
        selected_database_engine_state = gr.State()
        def update_selected_database_engine_state(db_engine: str):
            return db_engine

        ###################### SEARCH OPTIONS ######################
        features_choices_state = gr.State([])
        def update_features_choices_state(features_choices: List[str]):
            return features_choices
        vectors_choices_state = gr.State([])
        def update_vectors_choices_state(vector_choices: List[str]):
            return vector_choices

        ###################### SAVED DATABASE DROPDOWN ######################
        # STATE
        selected_database_state = gr.State()
        def update_selected_database_state(db_engine: str):
            return db_engine

        # Add state for tracking number of query pairs
        query_list_state = gr.State([1])
        added_count_state = gr.State(1)
        queries_state = gr.State([""])

        ###################### SEARCH TYPE ######################
        # STATE
        search_method_choice = gr.State()

        def update_search_method_choice(value: str):
            return value


        with gr.Row(
                variant='compact',
                equal_height=True
        ):
            # COMPONENT
            with gr.Column(
                scale=1
            ):
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

                    database_type_dropdown.change(
                        update_vectors_choices_state,
                        gr.State([]),
                        vectors_choices_state
                    )

                    database_type_dropdown.change(
                        update_features_choices_state,
                        gr.State([]),
                        features_choices_state
                    )

            with gr.Column(
                scale=6
            ):
                # COMPONENT
                @gr.render(inputs=[selected_database_engine_state])
                def create_saved_database_dropdown(selected_database_engine: str):
                    choices = []
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
                    saved_database_dropdown.change(
                        update_vectors_choices_state,
                        gr.State([]),
                        vectors_choices_state
                    )
                    saved_database_dropdown.change(
                        update_features_choices_state,
                        gr.State([]),
                        features_choices_state
                    )


        ###################### QUERY TEXTBOX ######################
        # query_input = gr.Textbox(
        #     label="🔎 Wpisz swoje pytanie"
        # )

        # Dynamic rendering of query textboxes and sliders with delete buttons
        @gr.render(inputs=[query_list_state])
        def render_query_inputs(query_list):
            def update_queries_state(*text_values):
                #print("Otrzymane teksty:", text_values)
                return list(text_values)

            def component_added(*text_values):
                list_of_values = list(text_values)
                list_of_values.append('')
                return list_of_values

            def component_deleted(queries_arg, index_arg):
                modified_query_list = list(queries_arg)
                del modified_query_list[index_arg]
                return modified_query_list


            count = len(query_list)
            text_boxes = []
            for i in range(count):
                with gr.Row(
                    equal_height=True
                ):
                    query_input = gr.Textbox(
                        scale=2,
                        label=f"🔎 Zapytanie {i + 1}",
                        key=f"query_input_{query_list[i]}"
                    )
                    query_input.input(
                        fn=update_queries_state,
                        inputs=text_boxes,
                        outputs=queries_state,
                        trigger_mode='always_last'
                    )

                    text_boxes.append(query_input)

                    if count > 1:
                        delete_btn = gr.Button(
                            "X",
                            size='sm',
                            min_width=20,
                            scale=0,
                            elem_classes="minimal-button"
                        )
                        delete_btn.click(
                            fn=remove_index,
                            inputs=[query_list_state, gr.State(i)],
                            outputs=query_list_state
                        ).then(
                            fn=component_deleted,
                            inputs=[queries_state, gr.State(i)],
                            outputs=queries_state
                        )

            add_query_btn.click(
                fn=add_query_component,
                inputs=[query_list_state, added_count_state],
                outputs=[query_list_state, added_count_state]
            ).then(
                fn=component_added,
                inputs=text_boxes,
                outputs=queries_state
            )


        # Add button to increase query count
        with gr.Row(
            equal_height=True
        ):
            ###################### TOP K SLIDER ######################
            top_k_slider = gr.Slider(
                1,
                100,
                10,
                step=1,
                label="🔝 Liczba najlepszych wyników",
                scale=9
            )
            add_query_btn = gr.Button("➕ Dodaj kolejne zapytanie", scale=1)

        def add_query_component(query_list: List, added_count):
            added_count = added_count + 1
            query_list.append(added_count)
            return query_list, added_count



        def remove_index(lst, idx):
            idx = int(idx)
            if 0 <= idx < len(lst):
                return lst[:idx] + lst[idx + 1:]
            return lst  # jeśli idx spoza zakresu, nic nie usuwamy



        # COMPONENT
        with gr.Row(
                equal_height=True
        ):
            @gr.render(inputs=[selected_database_engine_state, selected_database_state])
            def create_search_choices(database_type: str, vector_database_instance_json: str):
                if database_type and vector_database_instance_json:
                    choices: List[Tuple[str, str]] = []
                    vector_database_instance = VectorDatabaseInfo.from_dict(json.loads(vector_database_instance_json))

                    if DatabaseType.from_display_name(database_type).database_search:
                        choices.append((f'Native {database_type} search', database_type))


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

                    radio_buttons.change(
                        update_search_method_choice,
                        [radio_buttons],
                        [search_method_choice]
                    )

                    radio_buttons.change(
                        update_vectors_choices_state,
                        [gr.State([])],
                        [vectors_choices_state]
                    )

                    radio_buttons.change(
                        update_features_choices_state,
                        [gr.State([])],
                        [features_choices_state]
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

                    ):
                        embeddings_checkboxes = gr.CheckboxGroup(
                            label='Wybierz typy wektorów, które zostaną użyte do wyszukiwania wektorowego',
                            choices=choices
                        )

                        embeddings_checkboxes.change(
                            update_vectors_choices_state,
                            embeddings_checkboxes,
                            vectors_choices_state
                        )

                        features_choices = []
                        if search_type == DatabaseType.LANCE_DB and DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in vector_database_instance.features:
                            label_and_value = (f'{DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value} (use_tantivy={vector_database_instance.features[DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value]["use_tantivy"]})', DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value)
                            features_choices.append(label_and_value)

                        if features_choices:
                            features_checkboxes = gr.CheckboxGroup(
                                label='Inne typy wyszukiwania',
                                choices=features_choices
                            )

                            features_checkboxes.change(
                                update_features_choices_state,
                                features_checkboxes,
                                features_choices_state
                            )


        search_btn = gr.Button("🔍 Wyszukaj")

        search_output_state = gr.State([])

        search_btn.click(
            ui_search_database,
            [selected_database_engine_state, selected_database_state, queries_state, top_k_slider, search_method_choice,
             vectors_choices_state, features_choices_state],
            [search_output_state]
        )

        @gr.render(inputs=[search_output_state])
        def generate_tabs(search_output):
            for i in range(len(search_output)):
                print(f"{i}: {search_output[i]}")
                with gr.Tab(
                    label=f"Odpowiedz {i+1}"
                ):
                    text_display = gr.Textbox(
                        label="Wyniki wyszukiwania:",
                        interactive=False,
                        value=search_output[i],
                        lines=100
                    )

def ui_search_database(database_type: str, vector_database_instance_json: str, query_list: List[str], top_k: int, search_method: str, vector_choices: List[str], features_choices: List[str]):
    database_type_enum: DatabaseType = DatabaseType.from_display_name(database_type)
    vector_database_instance: VectorDatabaseInfo = database_type_enum.db_class.from_dict(json.loads(vector_database_instance_json))
    retrieved_text = perform_search(
        vector_database_instance=vector_database_instance,
        search_method=search_method,
        query_list=query_list,
        top_k=top_k,
        vector_choices=vector_choices,
        features_choices=features_choices
    )
    # token_count = count_tokens(retrieved_text)
    # characters_output = len(retrieved_text)
    # for i in range(len(retrieved_text)):
    #     print(f"{i}: {search_output}")
    return retrieved_text

# def ui_search_database(database_type: str, vector_database_instance_json: str, query_list: List[str], top_k: int, search_method: str, vector_choices: List[str], features_choices: List[str]):
#     database_type_enum: DatabaseType = DatabaseType.from_display_name(database_type)
#     vector_database_instance: VectorDatabaseInfo = database_type_enum.db_class.from_dict(json.loads(vector_database_instance_json))
#     retrieved_text = perform_search(
#         vector_database_instance=vector_database_instance,
#         search_method=search_method,
#         query_list=query_list,
#         top_k=top_k,
#         vector_choices=vector_choices,
#         features_choices=features_choices
#     )
#     token_count = count_tokens(retrieved_text)
#     characters_output = len(retrieved_text)
#     return token_count, characters_output, retrieved_text


def test_component(query_inputs_state, top_k_values_state):
    print(f'query: {query_inputs_state}')
    print(f'top_k: {top_k_values_state}')
