import gradio as gr
import json
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import list_cached_models, download_model_to_cache
from src.enums.embedding_type import EmbeddingType
import os

# Importujemy osobno utils dla Chroma i Lance
from src.chroma_db_utils import (
    create_new_database,
    get_databases_with_info_chroma_db
)
from src.lance_db_utils import (
    create_new_database_lance,
    get_databases_with_info_lance_db
)

from src.search_utils import (
    count_tokens,
    refresh_db_list
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
        db_list = get_databases_with_info_chroma_db()
    elif db_engine == "LanceDB":
        # LanceDB
        result = create_new_database_lance(db_name, files, chunk_size, chunk_overlap, model_name)
        # Odświeżamy listę baz Lance
        db_list = get_databases_with_info_lance_db()

    return result, gr.update(choices=db_list)

# Przy wyszukiwaniu musimy też zdecydować, czy to Chroma czy Lance
def ui_search_database(db_engine, db_name, query, top_k):
    print("CHROMA DB")
    if db_engine == "ChromaDB":
        from src.chroma_db_utils import retrieve_text_from_chroma_db
        retrieved_text = retrieve_text_from_chroma_db(db_name, query, top_k)
    elif db_engine == "LanceDB":
        from src.lance_db_utils import retrieve_text_from_lance_db
        retrieved_text = retrieve_text_from_lance_db(db_name, query, top_k)

    token_count = count_tokens(retrieved_text)
    return token_count, retrieved_text


def ui_add_model(model_name, selected_embedding_types):
    """
    Pobiera model z Hugging Face i zapisuje metadane o obsługiwanych typach embeddingów.
    """
    try:
        target_dir = download_model_to_cache(model_name)

        # Walidacja wartości – sprawdzamy, czy użytkownik nie podał błędnych opcji
        validated_embedding_types = [
            emb_type for emb_type in selected_embedding_types if emb_type in EmbeddingType.list()
        ]

        # Tworzymy plik metadata.json w folderze modelu
        metadata = {
            "model_name": model_name,
            "embedding_types": validated_embedding_types  # Zapisujemy tylko poprawne wartości
        }
        
        metadata_path = os.path.join(target_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Aktualizujemy listę modeli w cache
        new_model_list = list_cached_models()

        return (
            f"✅ Pomyślnie pobrano model '{model_name}'!\nFolder: {target_dir}",
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

            # Przycisk tworzenia nowej bazy
            create_db_btn = gr.Button("🛠️ Utwórz bazę")
            create_db_output = gr.Textbox(label="Wynik operacji")

            db_dropdown_create = gr.Dropdown(
                choices=[],
                label="📂 Wybierz bazę (Tworzenie nowej)"
            )

            create_db_btn.click(
                fn=ui_create_database,
                inputs=[db_engine_dropdown, db_name_input, file_uploader, chunk_size_slider, chunk_overlap_slider, model_dropdown],
                outputs=[create_db_output, db_dropdown_create]  # Używamy nowej zmiennej
            )

        ##################################################
        #   Zakładka 2: Wyszukiwanie w bazie
        ##################################################
        with gr.Tab("🔎 Wyszukiwanie w bazie"):
            # Najpierw dropdown do wyboru silnika
            search_engine_dropdown = gr.Dropdown(
                choices=["ChromaDB", "LanceDB"],
                value=None,
                label="Wybierz silnik wektorowy"
            )

            
            # Drugi dropdown z bazami – ale będzie dynamicznie aktualizowany
            db_dropdown_search = gr.Dropdown(
            choices=[],
            label="📂 Wybierz bazę (Wyszukiwanie)"
            )

            search_engine_dropdown.change(
                fn=refresh_db_list,
                inputs=search_engine_dropdown,
                outputs=db_dropdown_search  # ✅ Teraz poprawnie odwołujemy się do dropdown w wyszukiwaniu
            )

            query_input = gr.Textbox(label="🔎 Wpisz swoje pytanie")
            top_k_slider = gr.Slider(
                minimum=1, maximum=100, value=10, step=1, 
                label="🔝 Liczba najlepszych wyników"
            )
            search_btn = gr.Button("🔍 Szukaj")

            token_output = gr.Textbox(label="Liczba tokenów:", interactive=False)  
            search_output = gr.Textbox(label="Wyniki wyszukiwania:", interactive=False)  

            # Przycisk szukania
            search_btn.click(
                fn=ui_search_database, 
                inputs=[search_engine_dropdown, db_dropdown_search, query_input, top_k_slider], 
                outputs=[token_output, search_output]
        )

            

        ##################################################
        #   Zakładka 3: Dodawanie modelu do cache
        ##################################################
        with gr.Tab("🆕 Dodawanie modelu"):
            gr.Markdown("Wpisz nazwę modelu z Hugging Face i wybierz, jakie embeddingi obsługuje.")

            model_name_input_add = gr.Textbox(
                label="Wpisz nazwę modelu do pobrania:", 
                placeholder="Przykład: BAAI/bge-m3"
            )

            embedding_types = gr.CheckboxGroup(
                choices=EmbeddingType.list(),  # Pobieramy listę wartości enuma
                label="Wybierz obsługiwane typy embeddingów",
                value=[EmbeddingType.DENSE.value]  # Domyślnie zaznaczone Dense
            )

            add_model_btn = gr.Button("⬇️ Pobierz model do cache")
            add_model_output = gr.Textbox(label="Status dodawania modelu")

            # Obsługa kliknięcia
            add_model_btn.click(
                fn=ui_add_model, 
                inputs=[model_name_input_add, embedding_types],  
                outputs=[add_model_output, model_dropdown]
            )


        # --- LOGIKA AKCJI / CALLBACKI ---

        

        

        

        

    return app
