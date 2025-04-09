import traceback
from typing import List, Tuple

import gradio as gr
import json
from src.text_retrieval.models.downloaded_embedding_model import DownloadedEmbeddingModel
from src.ui.tabs.create_database_tab import create_downloaded_model_label


def add_model_tab(model_dropdown_choices_state):
    with gr.Tab("🆕 Dodawanie modelu"):
        gr.Markdown("Wpisz nazwę modelu z Hugging Face i wybierz, jakie embeddingi obsługuje.")

        model_name_input_add = gr.Textbox(
            label="Wpisz nazwę modelu do pobrania:",
            placeholder="Przykład: BAAI/bge-m3"
        )

        add_model_btn = gr.Button("⬇️ Pobierz model")
        add_model_output = gr.Textbox(label="Status dodawania modelu")


        add_model_btn.click(
            ui_add_model,
            [model_name_input_add, model_dropdown_choices_state],
            [add_model_output, model_dropdown_choices_state]
        )


def ui_add_model(model_name: str, model_dropdown_choices_state: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    try:
        model_instance: DownloadedEmbeddingModel = DownloadedEmbeddingModel.from_huggingface(model_name)
        list_of_model_instances = model_dropdown_choices_state
        list_of_model_instances.append((
            create_downloaded_model_label(model_instance),
            json.dumps(model_instance.to_dict(), ensure_ascii=False),
        ))
        return f"✅ Pomyślnie pobrano model '{model_name}'!", list_of_model_instances


    except Exception as e:
        traceback.print_exc()
        return f"❌ Błąd przy pobieraniu modelu '{model_name}': {e}", model_dropdown_choices_state
