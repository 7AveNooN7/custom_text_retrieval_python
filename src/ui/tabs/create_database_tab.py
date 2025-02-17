import json
from typing import List

import gradio as gr
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.db_utils import get_databases_with_info
from src.chroma_db_utils import create_new_database_chroma_db
from src.lance_db_utils import create_new_database_lance_db
from src.embedding_model_utils import get_downloaded_models
from src.enums.database_type import DatabaseType
from src.models.downloaded_model_info import DownloadedModelInfo
from src.models.vector_database_info import VectorDatabaseInfo


def ui_create_database(db_engine_from_dropdown: str, db_name_from_textbox: str, files_from_uploader: List, chunk_size_from_slider: int, chunk_overlap_from_slider: int, chosen_model_instance: DownloadedModelInfo):
    """Po naciśnięciu przycisku "Szukaj" zbiera dane z UI i tworzy nową bazę danych opartych na tych danych."""
    if not files_from_uploader or len(files_from_uploader) == 0:
        return "❌ Nie wybrano żadnego pliku!", gr.update(choices=[])

    if not db_engine_from_dropdown:
        return "❌ Nie wybrano embedding model!", gr.update(choices=[])
    else:
        db_engine_enum = DatabaseType(db_engine_from_dropdown)

    chosen_vector_database_info_instance = VectorDatabaseInfo(
        embedding_model_name=db_engine_from_dropdown,
        chunk_size=chunk_size_from_slider,
        chunk_overlap=chunk_overlap_from_slider,
        file_names=[file_obj.name for file_obj in files_from_uploader]
    )

    if db_engine_enum == DatabaseType.CHROMA_DB:
        result = create_new_database_chroma_db(db_name_from_textbox, chosen_vector_database_info_instance, chosen_model_instance)
    elif db_engine_enum == DatabaseType.LANCE_DB:
        result = create_new_database_lance_db(db_name_from_textbox, files_from_uploader, chunk_size_from_slider, chunk_overlap_from_slider, chosen_model_instance)

    db_list = get_databases_with_info(db_engine_enum)
    return result, gr.update(choices=db_list)


def create_database_tab():
    with gr.Tab("📂 Tworzenie nowej bazy"):
        db_engine_dropdown = gr.Dropdown(
            choices=[db.value for db in DatabaseType],
            value=DatabaseType.CHROMA_DB.value,
            label="Wybierz bazę wektorową"
        )
        db_name_input = gr.Textbox(label="🆕 Nazwa nowej bazy")
        file_uploader = gr.Files(
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
        create_db_output = gr.Textbox(label="Wynik operacji")


        def handle_create_db(db_engine_from_dropdown: str, db_name_from_textbox: str, files_from_uploader: List[gr.File], chunk_size_from_slider, chunk_overlap_from_slider, model_from_dropdown):
            model_json = json.loads(model_from_dropdown)  # 👉 Zamiana stringa JSON na słownik
            model_instance = DownloadedModelInfo.from_json(json_data=model_json)  # 👉 Przekazujemy poprawny format
            return ui_create_database(db_engine_from_dropdown, db_name_from_textbox, files_from_uploader, chunk_size_from_slider, chunk_overlap_from_slider, model_instance)

        create_db_btn.click(
            handle_create_db,  # 👈 Teraz przekazujemy funkcję zamiast `lambda`
            [
                db_engine_dropdown,
                db_name_input,
                file_uploader,
                chunk_size_slider,
                chunk_overlap_slider,
                model_dropdown  # 👈 To zwraca `str`, ale zamienimy go na instancję
            ],
            create_db_output
        )

    return model_dropdown
