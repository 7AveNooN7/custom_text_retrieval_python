import gradio as gr
from src.ui.tabs.create_database_tab import create_database_tab
from src.ui.tabs.search_database_tab import search_database_tab
from src.ui.tabs.add_model_tab import add_model_tab

def build_ui():
    with gr.Blocks() as app:
        gr.Markdown("# 🔍 Wyszukiwarka dokumentów i Zarządzanie Bazami")

        model_dropdown = create_database_tab()  # 📌 TERAZ PRZECHOWUJEMY ZWRÓCONY `model_dropdown`
        search_database_tab()
        add_model_tab(model_dropdown)  # 📌 PRZEKAZUJEMY `model_dropdown`
    return app