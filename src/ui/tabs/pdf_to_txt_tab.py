import asyncio
import os
from typing import Dict

import gradio as gr
from src.pdf_to_txt.pdf_to_txt_pipeline import PdfToTxtAnalysis, PdfFileInfo


def pdf_to_txt_tab():
    async def process_1(file_uploader_files_paths, files_state_arg):
        print(f'file_uploader_files_paths: {file_uploader_files_paths}')
        # JEZELI NIE MA ZADNEGO PLIKU TO NIC NIE WYSWIETLA
        if file_uploader_files_paths:
            ## gr.Files pozwala na duplikaty, ale ja nie, wiec usuwam duplikaty
            seen = set()
            file_uploader_files_paths = [x for x in file_uploader_files_paths if not (x in seen or seen.add(x))]

            # generuje liste ze sciezkami plikow ze wczesniejszymi plikami
            previous_path_list = []
            if files_state_arg:
                for file_name, file_value in files_state_arg.items():
                    previous_path_list.append(file_value.file_path)


            # jezeli z file picker cos sie usunelo to usuwam to tez ze stanu zeby sie nie wyswietlalo
            if len(previous_path_list) > len(file_uploader_files_paths):
                removed_files = [os.path.basename(path) for path in previous_path_list if path not in file_uploader_files_paths]
                for file in removed_files:
                    del files_state_arg[file]
                yield files_state_arg
                return

            # Jezeli dodano nowe pliki to tylko je trzeba przetworzyc
            paths_to_consider = [item for item in file_uploader_files_paths if item not in previous_path_list]


            pdf_to_txt_analysis = PdfToTxtAnalysis(
                path_list=paths_to_consider
            )

            # wysylam stan zeby pokazac processing
            temp_state_for_processing = {}
            for path in paths_to_consider:
                temp_state_for_processing[os.path.basename(path)] = None
            to_yield = {**files_state_arg, **temp_state_for_processing}
            yield to_yield  # DLA GRADIO ZEBY ZASSAL TEN YIELD
            await asyncio.sleep(0.3)


            new_pdf_files_info: Dict[str, PdfFileInfo] = pdf_to_txt_analysis.prepare_pdf_information()
            if isinstance(files_state_arg, dict):
                merged_pdf_files_info = {**files_state_arg, **new_pdf_files_info}
                yield merged_pdf_files_info
            else:
                yield new_pdf_files_info
        else:
            print('return {}')
            yield {}

    def print_something(value):
        print('OK')


    with gr.Tab("🔎 PDF to TXT"):
        # STATES
        files_state = gr.State({})

        with gr.Row(
            equal_height=True
        ):
            file_uploader = gr.Files(
                label="📤 Choose`.pdf` files to process",
                file_types=[".pdf"],
                scale=1
            )

            @gr.render(inputs=[files_state], triggers=[files_state.change], trigger_mode='multiple', queue=True)
            def test1(pdf_files):
                with gr.Column():
                    print(f'PDF_FILES: {pdf_files}')
                    if pdf_files:
                        for file_name, pdf_file_info in pdf_files.items():
                            with gr.Row(
                                equal_height=True
                            ):
                                text = gr.Text(value=file_name, show_label=False, scale=1)

                                if pdf_file_info is not None:
                                    conversion_method_radio = gr.Radio(
                                        scale=1,
                                        choices=["Simple", "Grobid"],
                                        show_label=False,
                                        info="Conversion method"
                                    )
                                    conversion_method_radio.value = 'Simple'

                                    filter_or_not_radio = gr.Radio(
                                        scale=1,
                                        choices=[("Yes", True), ("No", False)],
                                        show_label=False,
                                        info="Filter against EXCLUDED_TITLES",
                                        value=(pdf_file_info.filtered_toc is True),
                                        render=(pdf_file_info.filtered_toc is True)
                                    )

                                    empty_box = gr.Radio(
                                        scale=1,
                                        choices=[],
                                        show_label=False,
                                        render=(pdf_file_info.filtered_toc is not True)
                                    )

                                    conversion_method_radio.change(
                                        print_something,
                                        inputs=[conversion_method_radio],
                                        outputs=[]
                                    )
                                else:
                                    print(f'Create Processing Text ({file_name})')
                                    gr.HTML(value=processing_html)


            file_uploader.change(
                process_1,
                inputs=[file_uploader, files_state],
                outputs=[files_state]
            )


processing_html = """
<div style="display: flex; align-items: center; gap: 10px;">
    <div class="spinner" style="
        width: 20px;
        height: 20px;
        border: 3px solid #ccc;
        border-top: 3px solid #4A90E2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    "></div>
    <span style="font-weight: bold;">Processing...</span>
</div>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""