import gradio as gr
import json
import os

from src.config import MODEL_FOLDER
from src.embedding_model_utils import get_downloaded_models, download_model_from_hf
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary


def add_model_tab(model_dropdown):
    with gr.Tab("🆕 Dodawanie modelu"):
        gr.Markdown("Wpisz nazwę modelu z Hugging Face i wybierz, jakie embeddingi obsługuje.")

        model_name_input_add = gr.Textbox(
            label="Wpisz nazwę modelu do pobrania:",
            placeholder="Przykład: BAAI/bge-m3"
        )
        # embedding_types = gr.CheckboxGroup(
        #     choices=EmbeddingType.list(),
        #     label="Wybierz obsługiwane typy embeddingów",
        #     value=[EmbeddingType.DENSE.value]
        # )

        add_model_btn = gr.Button("⬇️ Pobierz model")
        add_model_output = gr.Textbox(label="Status dodawania modelu")


        add_model_btn.click(
            ui_add_model,
            [model_name_input_add],
            [add_model_output, model_dropdown]
        )


def ui_add_model(model_name):
    try:
        safe_model_dir: str = str(model_name.replace("/", "_"))
        target_dir: str = str(os.path.join(MODEL_FOLDER, safe_model_dir))
        download_model_from_hf(model_name, target_dir)

        # Pobranie informacji o wspieranych bibliotekach
        sentence_transformers_dict = TransformerLibrary.is_sentence_transformer_model(target_dir, model_name) or {}
        flag_embedding_dict = TransformerLibrary.is_flag_embedding_model(target_dir, model_name) or {}

        # Połączenie słowników
        supported_libraries = {**sentence_transformers_dict, **flag_embedding_dict}

        print(f"✅ supported_libraries: {supported_libraries}")

        # Tworzenie danych do zapisu w JSON
        metadata = {
            "model_name": model_name,
            "supported_libraries": supported_libraries
        }

        # Sprawdzenie poprawności przed zapisem
        metadata_path = os.path.join(target_dir, "metadata.json")
        try:
            json_str = json.dumps(metadata, indent=2, ensure_ascii=False)
            print(f'json_str: {json_str}')
        except TypeError as json_error:
            raise ValueError(f"Błąd przy konwersji do JSON: {json_error}")

        # Zapis do pliku metadata.json
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        print(f"✅ Plik metadata.json został zapisany: {metadata_path}")

        # Odświeżanie listy modeli
        new_model_list = get_downloaded_models()
        return f"✅ Pomyślnie pobrano model '{model_name}'!\nFolder: {target_dir}", gr.update(choices=new_model_list)

    except json.JSONDecodeError as json_error:
        return f"❌ Błąd przy generowaniu JSON: {json_error}", gr.update()

    except Exception as e:
        return f"❌ Błąd przy pobieraniu modelu '{model_name}': {e}", gr.update()
