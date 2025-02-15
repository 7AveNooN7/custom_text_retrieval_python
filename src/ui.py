import gradio as gr

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import list_cached_models, download_model_to_cache
from src.database_utils import (
    create_new_database, 
    get_databases_with_info
)
from src.search_utils import (
    retrieve_text, 
    count_tokens
)


def ui_create_database(db_name, files, chunk_size, chunk_overlap, model_name):
    """
    Callback do przycisku "Utwórz bazę".
    """
    if not files or len(files) == 0:
        return (
            "❌ Nie wybrano żadnego pliku!",
            gr.update(choices=get_databases_with_info())
        )

    result = create_new_database(db_name, files, chunk_size, chunk_overlap, model_name)
    # Odświeżamy dropdown
    return result, gr.update(choices=get_databases_with_info())


def ui_search_database(db_name, query, top_k):
    """
    Callback do przycisku "Szukaj".
    """
    retrieved_text = retrieve_text(db_name, query, top_k)
    token_count = count_tokens(retrieved_text)
    return token_count, retrieved_text


def ui_add_model(model_name):
    """
    Callback do pobierania modelu z Hugging Face do cache.
    """
    try:
        target_dir = download_model_to_cache(model_name)
        new_model_list = list_cached_models()
        return (
            f"✅ Pomyślnie pobrano model '{model_name}' do cache!\nFolder docelowy: {target_dir}",
            gr.update(choices=new_model_list)
        )
    except Exception as e:
        return (
            f"❌ Błąd przy pobieraniu modelu '{model_name}': {str(e)}",
            gr.update()
        )


def build_ui():
    """
    Główna funkcja budująca interfejs Gradio.
    Zwraca obiekt (Blocks), który można uruchomić w app.py
    """
    with gr.Blocks() as app:
        gr.Markdown("# 🔍 Wyszukiwarka dokumentów i Zarządzanie Bazami")

        ##################################################
        #   Zakładka 1: Tworzenie nowej bazy
        ##################################################
        with gr.Tab("📂 Tworzenie nowej bazy"):
            db_name_input = gr.Textbox(label="🆕 Nazwa nowej bazy")
            file_uploader = gr.Files(
                label="📤 Wybierz pliki `.txt` do przesłania:",
                file_types=[".txt"]
            )
            chunk_size_slider = gr.Slider(
                minimum=0, maximum=10000, value=DEFAULT_CHUNK_SIZE, step=100, 
                label="✂️ Długość fragmentu (liczba znaków)"
            )
            chunk_overlap_slider = gr.Slider(
                minimum=0, maximum=2000, value=DEFAULT_CHUNK_OVERLAP, step=50,
                label="🔄 Zachodzenie bloków (overlap)"
            )

            embedding_models = list_cached_models()
            model_dropdown = gr.Dropdown(
                choices=embedding_models, 
                value=embedding_models[0] if embedding_models else None, 
                label="🧠 Wybierz model embeddingowy (z cache)"
            )

            create_db_btn = gr.Button("🛠️ Utwórz bazę")
            create_db_output = gr.Textbox(label="Wynik operacji")

        ##################################################
        #   Zakładka 2: Wyszukiwanie w bazie
        ##################################################
        with gr.Tab("🔎 Wyszukiwanie w bazie"):
            db_dropdown = gr.Dropdown(
                choices=get_databases_with_info(),
                label="📂 Wybierz bazę wektorową"
            )

            query_input = gr.Textbox(label="🔎 Wpisz swoje pytanie")
            top_k_slider = gr.Slider(
                minimum=1, maximum=100, value=10, step=1, 
                label="🔝 Liczba najlepszych wyników"
            )
            search_btn = gr.Button("🔍 Szukaj")

            token_output = gr.Textbox(label="Liczba tokenów:", interactive=False)  
            search_output = gr.Textbox(label="Wyniki wyszukiwania:", interactive=False)  

        ##################################################
        #   Zakładka 3: Dodawanie modelu do cache
        ##################################################
        with gr.Tab("🆕 Dodawanie modelu"):
            gr.Markdown(
                "Wpisz nazwę modelu z Hugging Face (np. `BAAI/bge-m3`), aby go pobrać do lokalnego cache."
            )
            model_name_input_add = gr.Textbox(
                label="Nazwa modelu do pobrania", 
                placeholder="Przykład: BAAI/bge-m3"
            )
            add_model_btn = gr.Button("⬇️ Pobierz model do cache")
            add_model_output = gr.Textbox(label="Status dodawania modelu")

        # Logika przycisków i callbacki
        create_db_btn.click(
            fn=ui_create_database,
            inputs=[db_name_input, file_uploader, chunk_size_slider, chunk_overlap_slider, model_dropdown],
            outputs=[create_db_output, db_dropdown]
        )

        search_btn.click(
            fn=ui_search_database, 
            inputs=[db_dropdown, query_input, top_k_slider], 
            outputs=[token_output, search_output]
        )

        add_model_btn.click(
            fn=ui_add_model, 
            inputs=model_name_input_add,
            outputs=[add_model_output, model_dropdown]
        )

    return app
