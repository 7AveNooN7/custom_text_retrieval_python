
import gradio as gr
from src.pdf_to_txt.pdf_to_txt_pipeline import PdfToTxtAnalysis


def pdf_to_txt_tab():
    def process_1(file_uploader_files_paths):
        print(f'file_uploader_files: {file_uploader_files_paths}')

        pdf_to_txt_analysis = PdfToTxtAnalysis(
            path_list=file_uploader_files_paths,
            previous_path_list=[]
        )

        pdf_files_info = pdf_to_txt_analysis.prepare_pdf_information()

        print(pdf_files_info)


    with gr.Tab("🔎 PDF to TXT"):
        with gr.Row(
            equal_height=True
        ):
            file_uploader = gr.Files(
                label="📤 Choose`.pdf` files to process",
                file_types=[".pdf"],
                scale=1
            )

            analyse_textbox = gr.Textbox(
                label="Wstępna analiza .pdf"
            )

            file_uploader.change(
                process_1,
                inputs=[file_uploader],
                outputs=[analyse_textbox]
            )
