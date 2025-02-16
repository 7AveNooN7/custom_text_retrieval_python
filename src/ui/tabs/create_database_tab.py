import gradio as gr
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.db_utils import get_databases_with_info
from src.chroma_db_utils import create_new_database_chroma_db
from src.lance_db_utils import create_new_database_lance_db
from src.embeddings import list_cached_models
from src.enums.database_type import DatabaseType


def ui_create_database(db_engine: str, db_name, files, chunk_size, chunk_overlap, model_name):
    if not files or len(files) == 0:
        return "❌ Nie wybrano żadnego pliku!", gr.update(choices=[])

    db_engine_enum = DatabaseType(db_engine)

    if db_engine_enum == DatabaseType.CHROMA_DB:
        result = create_new_database_chroma_db(db_name, files, chunk_size, chunk_overlap, model_name)
    elif db_engine_enum == DatabaseType.LANCE_DB:
        result = create_new_database_lance_db(db_name, files, chunk_size, chunk_overlap, model_name)

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
        chunk_size_slider = gr.Slider(
            0,
            10000,
            DEFAULT_CHUNK_SIZE,
            step=100,
            label="✂️ Długość fragmentu"
        )
        chunk_overlap_slider = gr.Slider(
            0,
            2000,
            DEFAULT_CHUNK_OVERLAP,
            step=50,
            label="🔄 Zachodzenie bloków")

        embedding_models = list_cached_models()
        model_dropdown = gr.Dropdown(
            choices=embedding_models,
            value=embedding_models[0] if embedding_models else None,
            label="🧠 Model embeddingowy"
        )

        create_db_btn = gr.Button("🛠️ Utwórz bazę")
        create_db_output = gr.Textbox(label="Wynik operacji")

        create_db_btn.click(
            ui_create_database,
            [db_engine_dropdown, db_name_input, file_uploader, chunk_size_slider, chunk_overlap_slider,
            model_dropdown], create_db_output
        )

