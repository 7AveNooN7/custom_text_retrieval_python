# ui.py (fragment)

import gradio as gr

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import list_cached_models, download_model_to_cache

# Importujemy osobno utils dla Chroma i Lance
from src.chroma_db_utils import (
    create_new_database,
    get_databases_with_info
)
from src.lance_db_utils import (
    create_new_database_lance,
    get_databases_with_info_lance
)

from src.search_utils import (
    retrieve_text,
    count_tokens
)
# ... lub analogicznie "retrieve_text_lance" w innym miejscu ...

def ui_create_database(db_engine, db_name, files, chunk_size, chunk_overlap, model_name):
    """
    Callback do przycisku "Utwórz bazę".
    """
    if not files or len(files) == 0:
        return (
            "❌ Nie wybrano żadnego pliku!",
            gr.update(choices=[])  # zaktualizujemy dropdown baz
        )

    if db_engine == "ChromaDB":
        result = create_new_database(db_name, files, chunk_size, chunk_overlap, model_name)
        # Odświeżamy listę baz Chroma
        db_list = get_databases_with_info()
    elif db_engine == "LanceDB":
        # LanceDB
        result = create_new_database_lance(db_name, files, chunk_size, chunk_overlap, model_name)
        # Odświeżamy listę baz Lance
        db_list = get_databases_with_info_lance()

    return result, gr.update(choices=db_list)

# Przy wyszukiwaniu musimy też zdecydować, czy to Chroma czy Lance
def ui_search_database(db_engine, db_name, query, top_k):
    if db_engine == "ChromaDB":
        retrieved_text = retrieve_text(db_name, query, top_k)
    else:
        from src.lance_db_utils import retrieve_text_lance
        retrieved_text = retrieve_text_lance(db_name, query, top_k)

    token_count = count_tokens(retrieved_text)
    return token_count, retrieved_text


def ui_add_model(model_name):
    # bez zmian:
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
    with gr.Blocks() as app:
        gr.Markdown("# 🔍 Wyszukiwarka dokumentów i Zarządzanie Bazami")

        ##################################################
        #   Zakładka 1: Tworzenie nowej bazy
        ##################################################
        with gr.Tab("📂 Tworzenie nowej bazy"):
            # Nowe dropdown do wyboru silnika
            db_engine_dropdown = gr.Dropdown(
                choices=["ChromaDB", "LanceDB"],
                value="ChromaDB",
                label="Wybierz silnik wektorowy"
            )

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
            # Najpierw dropdown do wyboru silnika
            search_engine_dropdown = gr.Dropdown(
                choices=["ChromaDB", "LanceDB"],
                value="ChromaDB",
                label="Wybierz silnik wektorowy"
            )

            # Drugi dropdown z bazami – ale będzie dynamicznie aktualizowany
            db_dropdown = gr.Dropdown(
                choices=[],
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

        # --- LOGIKA AKCJI / CALLBACKI ---

        # Przycisk tworzenia nowej bazy
        create_db_btn.click(
            fn=ui_create_database,
            inputs=[db_engine_dropdown, db_name_input, file_uploader, chunk_size_slider, chunk_overlap_slider, model_dropdown],
            outputs=[create_db_output, db_dropdown]
        )

        # Kiedy zmienia się "search_engine_dropdown", musimy odświeżyć listę baz
        # zależnie od tego, czy to ChromaDB czy LanceDB:
        def refresh_db_list(engine_choice):
            if engine_choice == "ChromaDB":
                return gr.update(choices=get_databases_with_info())
            else:
                from src.lance_db_utils import get_databases_with_info_lance
                return gr.update(choices=get_databases_with_info_lance())

        search_engine_dropdown.change(
            fn=refresh_db_list,
            inputs=search_engine_dropdown,
            outputs=db_dropdown
        )

        # Przycisk szukania
        search_btn.click(
            fn=ui_search_database, 
            inputs=[search_engine_dropdown, db_dropdown, query_input, top_k_slider], 
            outputs=[token_output, search_output]
        )

        # Dodawanie modelu
        add_model_btn.click(
            fn=ui_add_model, 
            inputs=model_name_input_add,
            outputs=[add_model_output, model_dropdown]
        )

    return app
