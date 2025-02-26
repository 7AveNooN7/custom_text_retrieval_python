import asyncio
import json
import time
from typing import List, Literal, Tuple

import gradio as gr
from markdown_it.cli.parse import interactive
from pypika.enums import Boolean

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.db_utils import get_databases_with_info, is_valid_db_name
from src.chroma_db_utils import create_new_database_chroma_db
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary
from src.lance_db_utils import create_new_database_lance_db
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
        print(f'model_json: {model_json}')
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


def create_database_tab():
    with gr.Tab("📂 Tworzenie nowej bazy"):
        features_dict: dict = {}

        ###################### DATABASE TYPE DROPDOWN ######################
        selected_database_engine_state = gr.State()
        def change_selected_database_engine_state(db_engine: str):
            return db_engine
        @gr.render(inputs=[])
        def create_database_engine_dropdown():
            db_engine_dropdown = gr.Dropdown(
                choices=[db.display_name for db in DatabaseType],
                value=None,
                label="Wybierz bazę wektorową"
            )

            db_engine_dropdown.change(
                change_selected_library_state,
                [gr.State(None)],
                [selected_library_state]
            )

            db_engine_dropdown.change(
                change_selected_database_engine_state,
                [db_engine_dropdown],
                [selected_database_engine_state]
            )

            db_engine_dropdown.change(
                update_lance_db_fts_state,
                [gr.State(None), gr.State(None)],
                [lance_db_fts_state]
            )



        ###################### TEXTBOX ######################
        db_name_input = gr.Textbox(label="🆕 Nazwa nowej bazy")

        ###################### FILE UPLOADER ######################
        file_uploader = gr.Files(
            scale=50,
            label="📤 Wybierz pliki `.txt` do przesłania:",
            file_types=[".txt"]
        )

        ###################### MODEL DROPDOWN ######################
        model_dropdown_choices_state = gr.State(get_downloaded_models_for_dropdown())
        model_dropdown_current_choice_state = gr.State(None)
        def update_model_dropdown_current_choice(model_dropdown_current_choice_arg: str) -> str:
            return model_dropdown_current_choice_arg


        @gr.render(inputs=[model_dropdown_choices_state])
        def update_models_dropdown(model_dropdown_choices: List[Tuple[str, str]]):
            """
                1. embedding_models to lista instancji DownloadedModelInfo, funkcja get_downloaded_models() jest wywoływana przy starcie aplikacji
                2. Gradio Dropdown obsługuje tuple w taki sposób, że pokazuje tylko pierwszy element krotki.
                Przykład (wybrałem z listy GPT-4): ("GPT-4", '{"name": "GPT-4", "folder_name": "gpt4_model"}') -> dropdown do wyświetlania weźmie pierwszy element ("GPT-4") krotki.
                Ale, gdy przekaże jego wartość (model_dropdown) do innej funkcji to zwróci JSON ({"name": "GPT-4", "folder_name": "gpt4_model"}).
            """

            model_dropdown = gr.Dropdown(
                choices=model_dropdown_choices,  # Tuple (model_label, json)
                value=None,
                label="🧠 Model embeddingowy",
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

        ###################### RADIO BUTTONS ######################
        selected_library_state = gr.State()
        def change_selected_library_state(selected_library_state_arg):
            if selected_library_state_arg:
                return selected_library_state_arg
            else:
                return None


        @gr.render(inputs=[model_dropdown_current_choice_state, selected_database_engine_state])
        def generate_library_checkboxes(model_dropdown_current_choice: str, selected_database_engine_state: str):
            if model_dropdown_current_choice:
                model_instance = DownloadedModelInfo.from_dict(json_data=json.loads(model_dropdown_current_choice))
                radio = gr.Radio(
                    label='Wybierz bibliotekę do utworzenia embeddingów',
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
                    update_selected_choices,
                    [gr.State(None), gr.Number(value=0, visible=False)],
                    [selected_embeddings_state]
                )


        ###################### CHECKBOXES EMBEDDINGS ######################
        selected_embeddings_state = gr.State([])
        def update_selected_choices(choices: any, max_choices: int):
            if choices:
                if len(choices) > max_choices:
                    choices.pop(0)
                return choices
            else:
                return []

        @gr.render(inputs=[selected_database_engine_state, selected_embeddings_state, selected_library_state, model_dropdown_current_choice_state])
        def update_embedding_choices(db_engine_json: str, selected_embeddings: List[str], selected_library: str, model_instance_info: str):
            if db_engine_json and db_engine_json in [db.display_name for db in DatabaseType] and selected_library and model_dropdown_current_choice_state:
                db_type = DatabaseType.from_display_name(db_engine_json)
                embedding_choices_database = [et.value for et in db_type.storage_supported_embeddings]
                model_instance = DownloadedModelInfo.from_dict(json_data=json.loads(model_instance_info))
                model_info_supported_embeddings: List[EmbeddingType] = model_instance.get_supported_embeddings_from_specific_library(TransformerLibrary.from_display_name(selected_library))
                embedding_choices_model_info = [embedding.value for embedding in model_info_supported_embeddings]
                choices = list(set(embedding_choices_database) & set(embedding_choices_model_info))
                label = f"Wybierz jakie typy embeddingów utworzyć i zapisać w bazie (max: {db_type.simultaneous_embeddings})"
                # Tworzenie CheckboxGroup z aktualnymi opcjami
                checkbox_group = gr.CheckboxGroup(
                    choices=choices,
                    label=label,
                    value=selected_embeddings,
                    interactive=True,
                )

                max_choices_input = gr.Number(value=db_type.simultaneous_embeddings, visible=False)

                checkbox_group.change(
                    update_selected_choices,
                    inputs=[checkbox_group, max_choices_input],
                    outputs=selected_embeddings_state
                )

        ###################### LANCE DB FTS ######################
        lance_db_fts_state = gr.State({})
        def update_lance_db_fts_state(create_fts: Boolean, use_tantivy: str):
            main_key: str = DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value
            new_dict = {}
            if create_fts and use_tantivy:
                use_tantivy_boolean: Boolean = json.loads(use_tantivy)
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


        ###################### SLIDERS ######################
        chunk_size_slider = gr.Slider(
            1,
            1000000,
            DEFAULT_CHUNK_SIZE,
            step=100,
            label="✂️ Długość fragmentu"
        )
        chunk_overlap_slider = gr.Slider(
            1,
            500000,
            DEFAULT_CHUNK_OVERLAP,
            step=50,
            label="🔄 Zachodzenie bloków")

        create_db_btn = gr.Button("🛠️ Utwórz bazę")

        def handle_create_db(
                database_type_name: str,
                database_name_from_textbox: str,
                selected_embeddings: List[str],
                files_from_uploader: List[str],
                chunk_size_from_slider: int,
                chunk_overlap_from_slider: int,
                model_json_from_dropdown: str,
                selected_library: str
        ):
            yield gr.update(value="🚀 Tworzenie bazy danych!", interactive=False)

            # 2️⃣ Wywołujemy faktyczną operację
            ui_create_database(
                database_type_name,
                database_name_from_textbox,
                selected_embeddings,
                files_from_uploader,
                chunk_size_from_slider,
                chunk_overlap_from_slider,
                model_json_from_dropdown,
                selected_library,
                features_dict
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
                model_dropdown_current_choice_state,  # 👈 To zwraca `str`, ale zamienimy go na instancję
                selected_library_state
            ],
            [create_db_btn]
        )

    return model_dropdown_choices_state
