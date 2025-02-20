import gradio as gr
import json
import os
from src.embedding_model_utils import get_downloaded_models, download_model_from_hf
from src.enums.embedding_type_enum import EmbeddingType

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
        target_dir = download_model_from_hf(model_name)

        # Tworzymy plik metadata.json w folderze modelu
        metadata = {
            "model_name": model_name,
            "embedding_types": []  # Lista wybranych typów embeddingów
        }
        metadata_path = os.path.join(target_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # ODŚWIEŻANIE
        new_model_list = get_downloaded_models()
        return f"✅ Pomyślnie pobrano model '{model_name}'!\nFolder: {target_dir}", gr.update(choices=new_model_list)

    except Exception as e:
        return f"❌ Błąd przy pobieraniu modelu '{model_name}': {str(e)}", gr.update()
