import gradio as gr
from src.ui.tabs.create_database_tab import create_database_tab
from src.ui.tabs.search_database_tab import search_database_tab
from src.ui.tabs.add_model_tab import add_model_tab

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 Wyszukiwarka dokumentów i Zarządzanie Bazami")

        model_dropdown_choices = create_database_tab()  # 📌 TERAZ PRZECHOWUJEMY ZWRÓCONY `model_dropdown`
        search_database_tab()
        add_model_tab(model_dropdown_choices)  # 📌 PRZEKAZUJEMY `model_dropdown`
    return demo