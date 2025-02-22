from typing import List, Tuple

import gradio as gr
import json
import os

from src.config import MODEL_FOLDER
from src.embedding_model_utils import get_downloaded_models_for_dropdown, download_model_from_hf, add_model
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.downloaded_model_info import DownloadedModelInfo


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
    print(f'model_dropdown_choices_state: {model_dropdown_choices_state}')
    try:
        model_instance: DownloadedModelInfo = add_model(model_name)
        list_of_model_instances = model_dropdown_choices_state
        list_of_model_instances.append((
            model_instance.model_name,
            json.dumps(model_instance.to_dict(), ensure_ascii=False)
        ))
        return f"✅ Pomyślnie pobrano model '{model_name}'!", list_of_model_instances


    except Exception as e:
        return f"❌ Błąd przy pobieraniu modelu '{model_name}': {e}", model_dropdown_choices_state
