import json
from typing import List, Literal

import gradio as gr
from sympy import false

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.db_utils import get_databases_with_info, is_valid_db_name
from src.chroma_db_utils import create_new_database_chroma_db
from src.enums.embedding_type_enum import EmbeddingType
from src.lance_db_utils import create_new_database_lance_db
from src.embedding_model_utils import get_downloaded_models
from src.enums.database_type_enum import DatabaseType
from src.models.downloaded_model_info import DownloadedModelInfo


def ui_create_database(db_engine_from_dropdown: str, db_name_from_textbox: str, selected_choices_from_checkbox_group: List[str], files_from_uploader: List[str], chunk_size_from_slider: int, chunk_overlap_from_slider: int, chosen_model_instance: DownloadedModelInfo):
    """Po naciśnięciu przycisku "Szukaj" zbiera dane z UI i tworzy nową bazę danych opartych na tych danych."""
    # Walidacja nazwy bazy
    any_error = False


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

    if not db_engine_from_dropdown:
        gr.Warning(f"❌ Nie wybrano żadnego modelu do utworzenia embeddings!")
        any_error = True
    else:
        db_engine_enum = DatabaseType.from_display_name(db_engine_from_dropdown)

    if any_error:
        return None

    chosen_vector_database_info_instance = db_engine_enum.db_class(
        database_name=db_name_from_textbox,
        embedding_model_name=chosen_model_instance.model_name,
        chunk_size=chunk_size_from_slider,
        chunk_overlap=chunk_overlap_from_slider,
        files_paths=files_from_uploader,
        embedding_types=[EmbeddingType(choice) for choice in selected_choices_from_checkbox_group]
    )

    max_embeddings_count = chosen_vector_database_info_instance.get_database_type().simultaneous_embeddings
    if len(chosen_vector_database_info_instance.embedding_types) > max_embeddings_count:
        gr.Warning(f"❌ Wybrana baza danych nie obsługuje zapisania więcej embeddings niż {max_embeddings_count}!")
        return None

    print(f'files: {files_from_uploader}')

    if db_engine_enum == DatabaseType.CHROMA_DB:
        create_new_database_chroma_db(chosen_vector_database_info_instance)
    elif db_engine_enum == DatabaseType.LANCE_DB:
        create_new_database_lance_db(db_name_from_textbox, files_from_uploader, chunk_size_from_slider, chunk_overlap_from_slider, chosen_model_instance)


def create_database_tab():
    with gr.Tab("📂 Tworzenie nowej bazy"):
        db_engine_dropdown = gr.Dropdown(
            choices=[db.display_name for db in DatabaseType],
            value=None,
            label="Wybierz bazę wektorową"
        )

        db_name_input = gr.Textbox(label="🆕 Nazwa nowej bazy")

        file_uploader = gr.Files(
            scale=50,
            label="📤 Wybierz pliki `.txt` do przesłania:",
            file_types=[".txt"]
        )

        embedding_models = get_downloaded_models()
        """
            1. embedding_models to lista instancji DownloadedModelInfo, funkcja get_downloaded_models() jest wywoływana przy starcie aplikacji
            2. Gradio Dropdown obsługuje tuple w taki sposób, że pokazuje tylko pierwszy element krotki.
            Przykład (wybrałem z listy GPT-4): ("GPT-4", '{"name": "GPT-4", "folder_name": "gpt4_model"}') -> dropdown do wyświetlania weźmie pierwszy element ("GPT-4") krotki.
            Ale, gdy przekaże jego wartość (model_dropdown) do innej funkcji to zwróci JSON ({"name": "GPT-4", "folder_name": "gpt4_model"}).
        """
        model_choices = [(model.model_name, json.dumps(model.to_json())) for model in embedding_models]
        # Tworzymy dropdown, ale wartością jest cała instancja
        model_dropdown = gr.Dropdown(
            choices=model_choices,  # 👈 (nazwa, instancja)
            value=model_choices[0][1] if model_choices else None,  # Domyślna wartość: instancja
            label="🧠 Model embeddingowy"
        )

        selected_choices = gr.State([])

        def update_selected_choices(choices: List[str], max_choices: int):
            # print(f'choices: {choices}, max_choices: {max_choices}, len(choices): {len(choices)}')
            # if len(choices) > max_choices:
            #     print('popped')
            #     choices.pop(0)
            #     print(f'choices: {choices}')
            return choices


        # Główna funkcja renderująca
        @gr.render(inputs=[db_engine_dropdown])
        def update_embedding_choices(db_display_name):
            if db_display_name:
                db_type = DatabaseType.from_display_name(db_display_name)
                choices = [et.value for et in db_type.supported_embeddings]
                label = f"Wybierz jakie typy embeddingów utworzyć i zapisać w bazie (max: {db_type.simultaneous_embeddings})"
                # Tworzenie CheckboxGroup z aktualnymi opcjami
                checkbox_group = gr.CheckboxGroup(
                    choices=choices,
                    label=label,
                    #value=a,
                    interactive=True,
                    key="embedding_choices"  # Unikalny klucz dla komponentu
                )

                max_choices_input = gr.Number(value=db_type.simultaneous_embeddings, visible=False)

                checkbox_group.change(
                    update_selected_choices,
                    inputs=[checkbox_group, max_choices_input],
                    outputs=selected_choices
                )


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

        def handle_create_db(db_engine_from_dropdown: str, db_name_from_textbox: str, selected_choices_from_checkbox_group: List[str], files_from_uploader: List[str], chunk_size_from_slider: int, chunk_overlap_from_slider: int, model_from_dropdown: str):
            print(f'selected_choices_from_checkbox_group: {selected_choices_from_checkbox_group}')
            model_json = json.loads(model_from_dropdown)  # 👉 Zamiana stringa JSON na słownik
            model_instance = DownloadedModelInfo.from_json(json_data=model_json)  # 👉 Przekazujemy poprawny format
            return ui_create_database(db_engine_from_dropdown, db_name_from_textbox, selected_choices_from_checkbox_group, files_from_uploader, chunk_size_from_slider, chunk_overlap_from_slider, model_instance)

        create_db_btn.click(
            handle_create_db,  # 👈 Teraz przekazujemy funkcję zamiast `lambda`
            [
                db_engine_dropdown,
                db_name_input,
                selected_choices,
                file_uploader,
                chunk_size_slider,
                chunk_overlap_slider,
                model_dropdown  # 👈 To zwraca `str`, ale zamienimy go na instancję
            ],
            []
        )

    return model_dropdown
