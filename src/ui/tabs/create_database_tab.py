import json
import time
import gradio as gr
from typing import List, Tuple
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.db_utils import is_valid_db_name
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.floating_precision_enum import FloatPrecisionPointEnum
from src.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.enums.transformer_library_enum import TransformerLibrary
from src.embedding_model_utils import get_downloaded_models_for_dropdown
from src.enums.database_type_enum import DatabaseType, DatabaseFeature
from src.models.downloaded_model_info import DownloadedModelInfo
from src.save_to_database import save_to_database


def ui_create_database(
        db_engine_from_dropdown: str,
        db_name_from_textbox: str,
        selected_embeddings: List[str],
        files_from_uploader: List[str],
        chunk_size_from_slider: int,
        chunk_overlap_from_slider: int,
        model_json: str,
        selected_library: str,
        features_dict: dict
):
    """Po naciśnięciu przycisku "Szukaj" zbiera dane z UI i tworzy nową bazę danych opartych na tych danych."""
    # Walidacja nazwy bazy
    any_error = False
    if not db_engine_from_dropdown:
        gr.Warning(f"❌ Nie wybrano silnika bazy danych!")
        any_error = True
    else:
        db_engine_enum = DatabaseType.from_display_name(db_engine_from_dropdown)

    if not db_name_from_textbox:
        gr.Warning(f"❌ Nie podano nazwy bazy danych!")
        any_error = True
    else:
        if not is_valid_db_name(db_name_from_textbox):
            gr.Warning(f"❌ Niepoprawna nazwa bazy! Użyj tylko liter, cyfr, kropek i podkreśleń. Długość: 3-63 znaki.")
            any_error = True

    if not files_from_uploader or len(files_from_uploader) == 0:
        gr.Warning(f"❌ Nie wybrano żadnego pliku do utworzenia bazy danych!")
        any_error = True


    if not model_json:
        gr.Warning(f"❌ Nie wybrano żadnego modelu do utworzenia embeddings!")
        any_error = True
    else:
        model_instance = DownloadedModelInfo.from_dict(json_data=json.loads(model_json))


    if not selected_library:
        gr.Warning(f"❌ Nie wyrabo żadnego typu embeddingów!")
        any_error = True


    if any_error:
        return None

    chosen_vector_database_info_instance = db_engine_enum.db_class(
        database_name=db_name_from_textbox,
        embedding_model_name=model_instance.model_name,
        chunk_size=chunk_size_from_slider,
        chunk_overlap=chunk_overlap_from_slider,
        files_paths=files_from_uploader,
        embedding_types=[EmbeddingType(choice) for choice in selected_embeddings],
        transformer_library=TransformerLibrary.from_display_name(selected_library),
        features=features_dict
    )

    max_embeddings_count = chosen_vector_database_info_instance.get_database_type().simultaneous_embeddings
    if len(chosen_vector_database_info_instance.embedding_types) > max_embeddings_count:
        gr.Warning(f"❌ Wybrana baza danych nie obsługuje zapisania więcej embeddings niż {max_embeddings_count}!")
        return None

    saved_databases_dict: dict = chosen_vector_database_info_instance.get_saved_databases_from_drive_as_instances()
    saved_database_list = list(saved_databases_dict.keys())

    if db_name_from_textbox in saved_database_list:
        gr.Warning(f"❌ Już istnieje baza danych o takiej nazwie!")
        return None


    save_to_database(chosen_vector_database_info_instance)
    return None

text_segmentation_chose = None



def create_database_tab():
    with gr.Tab("📂 Create new database"):
        features_dict: dict = {}

        ###################### DATABASE ENGINE TYPE DROPDOWN ######################
        # STATE
        selected_database_engine_state = gr.State(None)
        def change_selected_database_engine_state(db_engine: str):
            return db_engine

        with gr.Row(
            variant='compact',
            equal_height=True
        ):
            # COMPONENT
            @gr.render(inputs=[])
            def create_database_engine_dropdown():

                db_engine_dropdown = gr.Dropdown(
                    choices=[db.display_name for db in DatabaseType],
                    value=None,
                    label="Select a database engine",
                    scale=1
                )

                # ZMIENIA STAN SAMEGO SIEBIE
                db_engine_dropdown.change(
                    change_selected_database_engine_state,
                    [db_engine_dropdown],
                    [selected_database_engine_state]
                )

                # ZMIENIA STAN LIBRARY RADIO NA NONE
                db_engine_dropdown.change(
                    change_selected_library_state,
                    [gr.State(None)],
                    [selected_library_state]
                )

                db_engine_dropdown.change(
                    update_lance_db_fts_state,
                    [gr.State(None), gr.State(None)],
                    [lance_db_fts_state]
                )


            ###################### TEXTBOX ######################
            db_name_input = gr.Textbox(label="🆕 New database name", scale=2)



        ###################### MODEL DROPDOWN ######################
        # STATE
        model_dropdown_choices_state = gr.State(get_downloaded_models_for_dropdown())
        model_dropdown_current_choice_state = gr.State(None)
        def update_model_dropdown_current_choice(model_dropdown_current_choice_arg: str) -> str:
            return model_dropdown_current_choice_arg

        # COMPONENT
        @gr.render(inputs=[model_dropdown_choices_state]) # TO SIE ZMIENIA TYLKO NA POCZATKU
        def create_models_dropdown(model_dropdown_choices: List[Tuple[str, str]]):
            """
                1. embedding_models to lista instancji DownloadedModelInfo, funkcja get_downloaded_models() jest wywoływana przy starcie aplikacji
                2. Gradio Dropdown obsługuje tuple w taki sposób, że pokazuje tylko pierwszy element krotki.
                Przykład (wybrałem z listy GPT-4): ("GPT-4", '{"name": "GPT-4", "folder_name": "gpt4_model"}') -> dropdown do wyświetlania weźmie pierwszy element ("GPT-4") krotki.
                Ale, gdy przekaże jego wartość (model_dropdown) do innej funkcji to zwróci JSON ({"name": "GPT-4", "folder_name": "gpt4_model"}).
            """
            with gr.Row(
                    variant='compact',
                    equal_height=True,
                    elem_id="custom_row"
            ):
                model_dropdown = gr.Dropdown(
                    choices=model_dropdown_choices,  # Tuple (model_label, json)
                    value=None,
                    label="🧠 Select an Embedding Model",
                    scale=2
                )

                radio = gr.Radio(
                    label='Select floating-point precision',
                    value=FloatPrecisionPointEnum.FP16.value,
                    choices=[fp.value for fp in FloatPrecisionPointEnum]
                )

            radio.change(
                update_text_segmentation_chose,
                [radio],
                [text_segmentation_chose_state]
            )

            model_dropdown.change(
                update_model_dropdown_current_choice,
                [model_dropdown],
                [model_dropdown_current_choice_state]
            )

            # PRZY ZMIANIE MODELU ZERUJE LIBRARY_STATE
            model_dropdown.change(
                change_selected_library_state,
                [gr.State(None)],
                [selected_library_state]
            )

        # LIBRARIES RADIO BUTTONS STATE
        selected_library_state = gr.State()
        def change_selected_library_state(selected_library_state_arg):
            if selected_library_state_arg:
                return selected_library_state_arg
            else:
                return None

        ###################### CHECKBOXES EMBEDDINGS ######################
        # STATE
        selected_embeddings_state = gr.State([])
        def update_selected_embeddings_choices(choices: any, max_choices: int):
            if choices:
                if len(choices) > max_choices:
                    choices.pop(0)
                return choices
            else:
                return []


        with gr.Row(
            equal_height=True,
            variant='compact',
            render=True
        ) as library_and_embeddings_row:
            # COMPONENT
            @gr.render(inputs=[model_dropdown_current_choice_state, selected_database_engine_state])
            def generate_library_checkboxes(model_dropdown_current_choice: str, selected_database_engine: str):
                if model_dropdown_current_choice and selected_database_engine:
                    model_instance = DownloadedModelInfo.from_dict(json_data=json.loads(model_dropdown_current_choice))
                    radio = gr.Radio(
                        label='Choose a python embedding library',
                        choices=[supported_library.display_name for supported_library, _ in
                                 model_instance.supported_libraries.items()],
                        value=None
                    )
                    radio.change(
                        change_selected_library_state,
                        [radio],
                        [selected_library_state]
                    )

                    radio.change(
                        update_selected_embeddings_choices,
                        [gr.State(None), gr.Number(value=0, visible=False)],
                        [selected_embeddings_state]
                    )
                else:
                    if not model_dropdown_current_choice and selected_database_engine:
                        gr.HTML(get_waiting_css_with_custom_text(
                            text='Awaiting the selection of an embedding model'), visible=True,render=True, padding=False)
                    elif model_dropdown_current_choice and not selected_database_engine:
                        gr.HTML(get_waiting_css_with_custom_text(
                            text='Awaiting the selection of a database engine'), visible=True, render=True,
                            padding=False)
                    elif not(model_dropdown_current_choice and selected_database_engine):
                        gr.HTML(get_waiting_css_with_custom_text(text='Awaiting the selection of a database engine and an embedding model'), visible=True, render=True, padding=False)


            # COMPONENT
            @gr.render(inputs=[selected_database_engine_state, selected_embeddings_state, selected_library_state, model_dropdown_current_choice_state])
            def create_embedding_choices(db_engine_json: str, selected_embeddings: List[str], selected_library: str, model_instance_info: str):
                if db_engine_json and db_engine_json in [db.display_name for db in DatabaseType] and selected_library and model_dropdown_current_choice_state:
                    db_type = DatabaseType.from_display_name(db_engine_json)
                    embedding_choices_database = [et.value for et in db_type.supported_embeddings]
                    model_instance = DownloadedModelInfo.from_dict(json_data=json.loads(model_instance_info))
                    model_info_supported_embeddings: List[EmbeddingType] = model_instance.get_supported_embeddings_from_specific_library(TransformerLibrary.from_display_name(selected_library))
                    embedding_choices_model_info = [embedding.value for embedding in model_info_supported_embeddings]
                    choices = list(set(embedding_choices_database) & set(embedding_choices_model_info))
                    label = f"Types of embedding vectors to create (max: {db_type.simultaneous_embeddings})"
                    # Tworzenie CheckboxGroup z aktualnymi opcjami
                    checkbox_group = gr.CheckboxGroup(
                        choices=choices,
                        label=label,
                        value=selected_embeddings,
                        interactive=True,
                    )

                    max_choices_input = gr.Number(value=db_type.simultaneous_embeddings, visible=False)

                    checkbox_group.change(
                        update_selected_embeddings_choices,
                        inputs=[checkbox_group, max_choices_input],
                        outputs=selected_embeddings_state
                    )
                else:
                    if not model_instance_info and db_engine_json:
                        gr.HTML(get_waiting_css_with_custom_text(
                            text='Awaiting the selection of an embedding model'), visible=True,render=True, padding=False)
                    elif model_instance_info and not db_engine_json:
                        gr.HTML(get_waiting_css_with_custom_text(
                            text='Awaiting the selection of a database engine'), visible=True, render=True,
                            padding=False)
                    elif not(model_instance_info and db_engine_json):
                        gr.HTML(get_waiting_css_with_custom_text(text='Awaiting the selection of a database engine and an embedding model'), visible=True, render=True, padding=False)
                    elif not selected_library:
                        gr.HTML(get_waiting_css_with_custom_text(
                            text='Awaiting the selection of an embedding library'), visible=True,
                                render=True, padding=False)




        ###################### LANCE DB FTS ######################
        # STATE
        lance_db_fts_state = gr.State({})
        def update_lance_db_fts_state(create_fts: bool, use_tantivy: str):
            main_key: str = DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value
            new_dict = {}
            if create_fts and use_tantivy:
                use_tantivy_boolean: bool = json.loads(use_tantivy)
                lance_db_features: dict = DatabaseType.LANCE_DB.features
                if create_fts:
                    new_dict = {
                        main_key: lance_db_features.get(main_key)
                    }
                    new_dict[main_key]["use_tantivy"] = use_tantivy_boolean
            if not new_dict:
                features_dict.pop(main_key, None)
            else:
                features_dict.update(new_dict)
            return new_dict

        # COMPONENT
        @gr.render(inputs=[selected_database_engine_state])
        def create_lance_db_fts(database_engine: str):
            if database_engine:
                database_type: DatabaseType = DatabaseType.from_display_name(database_engine)
                if database_type == DatabaseType.LANCE_DB:
                    database_features = database_type.features
                    if DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in database_features:
                        with gr.Row(
                            variant='compact',
                            equal_height=True
                        ):
                            checkbox = gr.Checkbox(
                                label="Create FTS index",
                                value=False,
                                info="Native LanceDB feature for creating BM25 index"
                            )

                            # Definiujemy opcje jako krotki (label, value)
                            radio_choices = [
                                ("Tantivy", json.dumps(True)),
                                ("Native LanceDB FTS", json.dumps(False))
                            ]

                            use_tantivy_from_enum = database_features[DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value]["use_tantivy"]
                            radio = gr.Radio(
                                show_label=False,
                                info='Use native LanceDB implementation or Tantivy',
                                value=json.dumps(use_tantivy_from_enum),
                                choices=radio_choices
                            )

                            radio.change(
                                update_lance_db_fts_state,
                                [checkbox, radio],
                                lance_db_fts_state
                            )

                            checkbox.change(
                                update_lance_db_fts_state,
                                [checkbox, radio],
                                lance_db_fts_state
                            )
                else:
                    #gr.HTML("create_lance_db_fts", visible=True, render=False, padding=False)
                    return None
            else:
                #gr.HTML("create_lance_db_fts", visible=True, render=False, padding=False)
                return None


        ###################### SLIDERS ######################
        with gr.Row(
            equal_height=True,
            variant= 'compact'
        ):
            ###################### FILE UPLOADER ######################
            file_uploader = gr.Files(
                label="📤 Choose`.txt` files to process",
                file_types=[".txt"],
                scale=1
            )
            with gr.Column():
                chunk_size_slider = gr.Slider(
                    1,
                    1000000,
                    DEFAULT_CHUNK_SIZE,
                    step=100,
                    label="📏 Chunk length"
                )
                chunk_overlap_slider = gr.Slider(
                    1,
                    500000,
                    DEFAULT_CHUNK_OVERLAP,
                    step=50,
                    label="🔄 Chunk overlap")

                segmentation_type_radio = gr.Radio(
                    label='✂️ Select text segmentation type',
                    value=TextSegmentationTypeEnum.TOKENS.value,
                    choices=[ts.value for ts in TextSegmentationTypeEnum],
                    interactive=True
                )


        ###################### TEXT SEGMENTATION OPTIONS ######################
        # STATE
        text_segmentation_chose_state = gr.State(FloatPrecisionPointEnum.FP16.value)
        def update_text_segmentation_chose(choice: str):
            return choice



        create_db_btn = gr.Button("🛠️ Create a new database")

        def handle_create_db(
                database_type_name: str,
                database_name_from_textbox: str,
                selected_embeddings: List[str],
                files_from_uploader: List[str],
                chunk_size_from_slider: int,
                chunk_overlap_from_slider: int,
                model_json_from_dropdown: str,
                selected_library: str,
                text_segmentation_chose: str,
        ):
            yield gr.update(value="🚀 Tworzenie bazy danych!", interactive=False)


            print(f'selected_embeddings: {selected_embeddings}')
            print(f'text_segmentation_chose_state: {text_segmentation_chose_state}')

            # Wywołujemy faktyczną operację
            ui_create_database(
                database_type_name,
                database_name_from_textbox,
                selected_embeddings,
                files_from_uploader,
                chunk_size_from_slider,
                chunk_overlap_from_slider,
                model_json_from_dropdown,
                selected_library,
                features_dict  #Zmienna "globalna"
            )

            # 3️⃣ Zmieniamy tekst na "CHUJ" po zakończeniu operacji
            yield gr.update(value="✅ Baza danych została utworzona!", interactive=False)
            time.sleep(2)
            yield gr.update(value="🛠️ Utwórz nową bazę", interactive=True)


        create_db_btn.click(
            handle_create_db,  # 👈 Teraz przekazujemy funkcję zamiast `lambda`
            [
                selected_database_engine_state,
                db_name_input,
                selected_embeddings_state,
                file_uploader,
                chunk_size_slider,
                chunk_overlap_slider,
                model_dropdown_current_choice_state,
                selected_library_state,
                text_segmentation_chose_state
            ],
            [create_db_btn]
        )

    return model_dropdown_choices_state

def get_waiting_css_with_custom_text(*, text):
    return f"""
<div style='display: flex; justify-content: center; align-items: center; padding: 10px; width: 100%; height: 100%;'>
    <div style='border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; margin-right: 10px;'></div>
    <span style='color: #666; font-style: italic;'>{text}...</span>
</div>
<style>
@keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}
</style>
""".strip()