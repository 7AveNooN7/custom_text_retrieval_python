import asyncio
import os
from multiprocessing import Manager
from typing import Dict

import gradio as gr
import copy

from src.config import GROBID_MAX_WORKERS
from src.pdf_to_txt.convert_files import ConvertFiles
from src.pdf_to_txt.models.file_settings_model import FileSettingsModel, ConversionMethodEnum
from src.pdf_to_txt.pdf_file_info import PdfToTxtAnalysis, PdfFileInfo
from src.ui.tabs.create_database_tab import get_waiting_css_with_custom_text


def pdf_to_txt_tab():
    async def process_1(file_uploader_files_paths, files_state_arg):
        #print(f'file_uploader_files_paths: {file_uploader_files_paths}')
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
            yield to_yield
            await asyncio.sleep(0.3) # DLA GRADIO ZEBY ZASSAL TEN YIELD


            new_pdf_files_info: Dict[str, PdfFileInfo] = pdf_to_txt_analysis.prepare_pdf_information()
            if isinstance(files_state_arg, dict):
                merged_pdf_files_info = {**files_state_arg, **new_pdf_files_info}
                yield merged_pdf_files_info
            else:
                yield new_pdf_files_info
        else:
            #print('return {}')
            yield {}

    # GENERALNIE DALEM KLUCZE DO KOMPONENTOW WIEC WARTOSCI RADIO BEDA ZAPAMIETANE NA ZAWSZE WIEC NIE MUSZE SIE PRZEJMOWAC USUWANIEM
    def initialize_file_settings_state(file_uploader_files_paths, files_settings_state_arg):
        # ZBIERAM LISTE PLIKOW KTORE MAJA JUZ SETTINGS
        if file_uploader_files_paths:
            #print(f'file_uploader_files_paths: {file_uploader_files_paths}')
            file_settings_state_files = []
            for file_name, file_settings_value in files_settings_state_arg.items():
                file_settings_state_files.append(file_name)

            #print(f'file_settings_state_files: {file_settings_state_files}')

            # ZBIERAM LISTE PLIKOW KTORE NIE BYLY JUZ NIGDY UPLOADOWANE
            difference = [os.path.basename(f) for f in file_uploader_files_paths if os.path.basename(f) not in file_settings_state_files]
            #print(f"diff: {difference}")
            new_files_dict = {}
            for file_name in difference:
                new_files_dict[file_name] = FileSettingsModel(
                    conversion_method=ConversionMethodEnum.SIMPLE,
                    use_filter=False
                )

            return {**files_settings_state_arg, **new_files_dict}
        else:
            return files_settings_state_arg


    def update_file_settings_state(files_settings_arg: Dict[str, FileSettingsModel], file_name: str, radio_chose):
        if isinstance(radio_chose, bool):
            files_settings_arg[file_name].use_filter = radio_chose

        if ConversionMethodEnum.is_enum_value(radio_chose):
            files_settings_arg[file_name].conversion_method = ConversionMethodEnum(radio_chose)

        print(f'file_name: {file_name}, radio_chose: {radio_chose}')
        return files_settings_arg


    with gr.Tab("🔎 PDF to TXT"):
        # STATES
        files_state = gr.State({})
        files_settings_state = gr.State({})

        interactive_state = gr.State(True)
        def update_interactive_state(state: bool):
            return state



        with gr.Row(
            equal_height=True
        ):
            with gr.Column(
                scale=1
            ):
                file_uploader = gr.Files(
                    label="📤 Choose`.pdf` files to process",
                    file_types=[".pdf"],
                    scale=1,
                    visible=True
                )
                database_text = gr.Text(label="Nazwa folderu:", visible=True)


            file_uploader.change(
                initialize_file_settings_state,
                inputs=[file_uploader, files_settings_state],
                outputs=[files_settings_state]
            ).then(
                process_1,
                inputs=[file_uploader, files_state],
                outputs=[files_state]
            )
            with gr.Column(
                    scale=2
            ):
                @gr.render(inputs=[files_state, files_settings_state, interactive_state], triggers=[files_state.change, interactive_state.change], trigger_mode='multiple', queue=True)
                def test1(pdf_files, files_settings_arg, interactive_state_arg):

                        #print(f'PDF_FILES: {pdf_files}')
                        if pdf_files:
                            for file_name, pdf_file_info in pdf_files.items():
                                with gr.Row(
                                    equal_height=True,
                                    variant="compact"
                                ):
                                    text = gr.Text(value=file_name, show_label=False, scale=1, key=f"{file_name}_key")
                                    print(f'interactive_state_arg1: {interactive_state_arg}')

                                    if pdf_file_info is not None:
                                        conversion_method_radio = gr.Radio(
                                            scale=2,
                                            choices=[
                                                ConversionMethodEnum.SIMPLE.value,
                                                ConversionMethodEnum.GROBID.value,
                                            ],
                                            value=files_settings_arg[file_name].conversion_method,
                                            show_label=False,
                                            interactive=interactive_state_arg,
                                            info="Conversion method",
                                            key=f"{file_name}_conversion_method_radio_key"
                                        )

                                        conversion_method_radio.change(
                                            update_file_settings_state,
                                            inputs=[files_settings_state, gr.State(file_name), conversion_method_radio],
                                            outputs=[files_settings_state]
                                        )

                                        filter_or_not_radio = gr.Radio(
                                            scale=2,
                                            choices=[("Yes", True), ("No", False)],
                                            show_label=False,
                                            interactive=interactive_state_arg,
                                            info="Filter against EXCLUDED_TITLES",
                                            key=f"{file_name}_filter_or_not_radio_key",
                                            value=files_settings_arg[file_name].use_filter,
                                            render=(pdf_file_info.filtered_toc is True),
                                        )

                                        filter_or_not_radio.change(
                                            update_file_settings_state,
                                            inputs=[files_settings_state, gr.State(file_name), filter_or_not_radio],
                                            outputs=[files_settings_state]
                                        )

                                        empty_box = gr.Radio(
                                            scale=2,
                                            choices=[],
                                            show_label=False,
                                            render=(pdf_file_info.filtered_toc is not True)
                                        )

                                        # if not interactive_state_arg:
                                        #     gr.HTML(value=get_waiting_css_with_custom_text(text=""))

                                    else:
                                        gr.HTML(value=get_waiting_css_with_custom_text(text="Analysis"))


        with gr.Column():
            circular_progress = gr.HTML(value=get_waiting_css_with_custom_text(text="Processing .pdf files..."), visible=False)
            convert_button = gr.Button(value='Convert')


        def test_2(files_state_arg, files_settings_state_arg, database_folder_name: str):
            yield get_waiting_css_with_custom_text(text="Processing .pdf files...")
            manager = Manager()
            grobid_semaphore = manager.Semaphore(GROBID_MAX_WORKERS)
            convert_files = ConvertFiles(
                files_state=copy.deepcopy(files_state_arg),
                files_settings_state=copy.deepcopy(files_settings_state_arg),
                database_folder_name=database_folder_name,
                grobid_semaphore=grobid_semaphore
            )

            for komunikat in convert_files.start_converting_files():
                yield komunikat

        convert_button.click(
            fn=lambda: (
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),  # albo gr.update() jeśli nie chcesz nic zmieniać
                gr.update(visible=True),
                gr.update(visible=False)
            ),
            inputs=[],
            outputs=[convert_button, file_uploader, circular_progress, database_text]

        ).then(
            fn=update_interactive_state,
            inputs=[gr.State(False)],
            outputs=[interactive_state]
        ).then(
            fn=test_2,
            inputs=[files_state, files_settings_state, database_text],
            outputs=[circular_progress]
        ).then(
            fn=update_interactive_state,
            inputs=[gr.State(True)],
            outputs=[interactive_state]
        ).then(
            fn=lambda: (
                gr.update(interactive=True, visible=True),
                gr.update(interactive=True, visible=True),  # albo gr.update() jeśli nie chcesz nic zmieniać
                gr.update(visible=False),
                gr.update(visible=True)
            ),
            inputs=[],
            outputs=[convert_button, file_uploader, circular_progress, database_text]
        )

