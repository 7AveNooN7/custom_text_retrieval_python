from typing import Dict

import gradio as gr
from src.pdf_to_txt.pdf_to_txt_pipeline import PdfToTxtAnalysis, PdfFileInfo


def pdf_to_txt_tab():
    def process_1(file_uploader_files_paths):


        if file_uploader_files_paths:
            print(f'file_uploader_files: {file_uploader_files_paths}')

            pdf_to_txt_analysis = PdfToTxtAnalysis(
                path_list=file_uploader_files_paths,
                previous_path_list=[]
            )

            pdf_files_info: Dict[str, PdfFileInfo] = pdf_to_txt_analysis.prepare_pdf_information()

            return pdf_files_info
        else:
            return {}

    def print_something(value):
        print('OK')


    with gr.Tab("🔎 PDF to TXT"):
        # STATES
        files_state = gr.State()

        with gr.Row(
            equal_height=True
        ):
            file_uploader = gr.Files(
                label="📤 Choose`.pdf` files to process",
                file_types=[".pdf"],
                scale=1
            )

            @gr.render(inputs=[files_state])
            def test1(pdf_files):
                with gr.Column():
                    if pdf_files:
                        for file_name, pdf_file_info in pdf_files.items():
                            with gr.Row(
                                equal_height=True
                            ):
                                text = gr.Text(value=file_name, show_label=False)
                                conversion_method_radio = gr.Radio(choices=["Simple", "Grobid"], show_label=False, info="Conversion method")
                                conversion_method_radio.value = 'Simple'

                                filter_or_not_radio = gr.Radio(
                                    choices=[("Yes", True), ("No", False)],
                                    show_label=False,
                                    info="Filter against EXCLUDED_TITLES",
                                    value=(pdf_file_info.filtered_toc is True),
                                    render=(pdf_file_info.filtered_toc is True)
                                )

                                conversion_method_radio.change(
                                    print_something,
                                    inputs=[conversion_method_radio],
                                    outputs=[]
                                )
                    else:
                        gr.Text(value='Oczekiwanie na wybor plikow', show_label=False)


            file_uploader.change(
                process_1,
                inputs=[file_uploader],
                outputs=[files_state]
            )
