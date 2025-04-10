import os
from typing import Dict

import gradio as gr
from src.pdf_to_txt.pdf_to_txt_pipeline import PdfToTxtAnalysis, PdfFileInfo


def pdf_to_txt_tab():
    def process_1(file_uploader_files_paths, files_state_arg):
        if file_uploader_files_paths:
            previous_path_list = []
            if files_state_arg:
                for file_name, file_value in files_state_arg.items():
                    previous_path_list.append(file_value.file_path)

            print(f'previous_path_list: {previous_path_list}')


            if len(previous_path_list) > len(file_uploader_files_paths):
                removed_files = [os.path.basename(path) for path in previous_path_list if path not in file_uploader_files_paths]
                for file in removed_files:
                    del files_state_arg[file]
                return files_state_arg

            print(f'file_uploader_files: {file_uploader_files_paths}')
            paths_to_consider = [item for item in file_uploader_files_paths if item not in previous_path_list]

            pdf_to_txt_analysis = PdfToTxtAnalysis(
                path_list=paths_to_consider
            )

            new_pdf_files_info: Dict[str, PdfFileInfo] = pdf_to_txt_analysis.prepare_pdf_information()
            print(f'new_pdf_files_info: {[info.file_name for info in new_pdf_files_info.values()]}')
            if isinstance(files_state_arg, dict):
                merged_pdf_files_info = {**files_state_arg, **new_pdf_files_info}
                print(f'merged_pdf_files_info: {[info.file_name for info in merged_pdf_files_info.values()]}')
                return merged_pdf_files_info
            else:
                return new_pdf_files_info
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

            @gr.render(inputs=[files_state], triggers=[files_state.change])
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
                inputs=[file_uploader, files_state],
                outputs=[files_state]
            )
