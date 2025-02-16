import gradio as gr
import json
import os
from src.embeddings import list_cached_models, download_model_to_cache
from src.enums.embedding_type import EmbeddingType

def add_model_tab():
    with gr.Tab("🆕 Dodawanie modelu"):
        gr.Markdown("Wpisz nazwę modelu z Hugging Face i wybierz, jakie embeddingi obsługuje.")

        model_name_input_add = gr.Textbox(
            label="Wpisz nazwę modelu do pobrania:",
            placeholder="Przykład: BAAI/bge-m3"
        )
        embedding_types = gr.CheckboxGroup(
            choices=EmbeddingType.list(),
            label="Wybierz obsługiwane typy embeddingów",
            value=[EmbeddingType.DENSE.value]
        )

        add_model_btn = gr.Button("⬇️ Pobierz model do cache")
        add_model_output = gr.Textbox(label="Status dodawania modelu")

        add_model_btn.click(ui_add_model, [model_name_input_add, embedding_types], [add_model_output])


def ui_add_model(model_name, selected_embedding_types):
    try:
        target_dir = download_model_to_cache(model_name)

        validated_embedding_types = [emb_type for emb_type in selected_embedding_types if
                                     emb_type in EmbeddingType.list()]

        metadata = {"model_name": model_name, "embedding_types": validated_embedding_types}

        metadata_path = os.path.join(target_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        new_model_list = list_cached_models()
        return f"✅ Pomyślnie pobrano model '{model_name}'!\nFolder: {target_dir}", gr.update(choices=new_model_list)

    except Exception as e:
        return f"❌ Błąd przy pobieraniu modelu '{model_name}': {str(e)}", gr.update()
