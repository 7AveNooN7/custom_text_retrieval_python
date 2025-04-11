import os
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Tuple
from datetime import datetime
from pymupdf import pymupdf

from src.config import TXT_FOLDER, GROBID_URL
from src.pdf_to_txt.models.file_settings_model import FileSettingsModel, ConversionMethodEnum
from src.pdf_to_txt.pdf_file_info import PdfFileInfo


class ConvertFiles:
    def __init__(self, files_state: Dict[str, PdfFileInfo], files_settings_state: Dict[str, FileSettingsModel], database_folder_name: str):
        self.files_state: Dict[str, PdfFileInfo] = files_state
        self.files_settings_state: Dict[str, FileSettingsModel] = files_settings_state
        self.database_folder_name: str = database_folder_name

    def create_text_folder_path(self):
        now = datetime.now()
        folder_name_time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        database_folder_name = self.database_folder_name.strip()

        if database_folder_name:
            database_folder_name = database_folder_name + f' ({folder_name_time_stamp})'
        else:
            database_folder_name = folder_name_time_stamp

        database_folder_path = os.path.join(TXT_FOLDER, database_folder_name)

        os.makedirs(database_folder_path)

        return database_folder_path


    def start_converting_files(self):
        txt_folder_path = self.create_text_folder_path()

        # Prepare arguments for each file to be processed
        tasks = [
            (file_name, file_value, self.files_settings_state[file_name], txt_folder_path)
            for file_name, file_value in self.files_state.items()
        ]

        # Use multiprocessing Pool to process files in parallel
        with Pool() as pool:
            pool.map(self._convert_single_file, tasks)

    def _convert_single_file(self, args: Tuple[str, PdfFileInfo, FileSettingsModel, str]) -> None:
        """
        Worker function to convert a single file. This will run in a separate process.

        Args:
            args: Tuple containing (file_name, file_value, file_settings, txt_folder_path)
        """
        file_name, file_value, file_settings, txt_folder_path = args

        if file_settings.conversion_method == ConversionMethodEnum.SIMPLE:
            text_to_save = self.simple_method(file=file_value, file_settings=file_settings)
            output_path = os.path.join(txt_folder_path, f"{file_name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_to_save)

        elif file_settings.conversion_method == ConversionMethodEnum.GROBID:
            text_to_save = self.grobid_method(file=file_value, file_settings=file_settings, folder_path=txt_folder_path)
            output_path = os.path.join(txt_folder_path, f"{file_name}.txt")
            # with open(output_path, "w", encoding="utf-8") as f:
            #     f.write(text_to_save)


    def simple_method(self, file: PdfFileInfo, file_settings: FileSettingsModel) -> str:
        doc: pymupdf.Document = pymupdf.open(file.file_path)

        start_page = file.start_page
        end_page = file.end_page

        if file_settings.use_filter and file.filtered_toc:
            start_page = file.filtered_start_page
            end_page = file.filtered_end_page

        text = "\n".join(
            doc[page_num].get_text().strip()
            for page_num in range(start_page - 1, end_page)
        )
        return text

    def grobid_method(self, file: PdfFileInfo, file_settings: FileSettingsModel, folder_path: str) -> str:
        # print(f'file_name: {file.file_name}')
        # print('chapters:')
        # for chapter_name, chapter_value in file.chapter_info.items():
        #     print(f'{chapter_name}: {chapter_value.start_page} - {chapter_value.end_page}')
        # print('synthetic chapters:')
        # for chapter_name, chapter_value in file.synthetic_chapter_info.items():
        #     print(f'{chapter_name}: {chapter_value.start_page} - {chapter_value.end_page}')

        # Tworzy folder do przechowywania podzielonycn pdf i txt dla aktualnie przetwarzanego pliku .pdf
        current_file_folder_path = os.path.join(folder_path, Path(file.file_name).stem.replace(" ", "_"))
        os.makedirs(current_file_folder_path)

        # tworzy podzielone .pdf dla grobid do przetowrzenia
        self.create_split_pdf_files(file=file, file_settings=file_settings, folder_path=current_file_folder_path)

        self.process_files_with_grobid(current_file_folder_path=current_file_folder_path)


    def create_split_pdf_files(self, file: PdfFileInfo, file_settings: FileSettingsModel, folder_path: str) -> None:
        doc: pymupdf.Document = pymupdf.open(file.file_path)

        chapters_to_iterate_for = file.chapter_info if (file_settings.use_filter and file.filtered_toc) else file.filtered_chapter_info

        for chapter_name, chapter_info_value in chapters_to_iterate_for.items():
            new_doc = pymupdf.open()
            new_doc.insert_pdf(doc, from_page=chapter_info_value.start_page - 1,
                               to_page=chapter_info_value.end_page - 1)
            new_doc.save(os.path.join(folder_path, f"{file.file_name.replace(" ", "_")}_{chapter_name.replace(" ", "_")}.pdf"))
            new_doc.close()


    def process_files_with_grobid(self, current_file_folder_path: str):
        # lista wszystkich pelnych sciezek plikow .pdf
        pdf_files = [os.path.join(current_file_folder_path, f) for f in os.listdir(current_file_folder_path) if f.endswith(".pdf")]

        with ThreadPoolExecutor() as executor:
            # Tworzymy przyszłe zadania dla każdego pliku PDF
            futures = {executor.submit(self.process_with_grobid, pdf, current_file_folder_path): pdf for pdf in pdf_files}


    def process_with_grobid(self, pdf_full_path: str, current_file_folder_path: str):
        xml_file_name = Path(pdf_full_path).stem + ".xml"
        xml_output_path = os.path.join(current_file_folder_path, xml_file_name)

        with open(pdf_full_path, "rb") as pdf:
            files = {"input": pdf}

            data = {
                "consolidateHeader": "0",
                "consolidateCitations": "0",
                "consolidateFunders": "0",
                "includeRawCitations": "0",
                "includeRawAffiliations": "0",
                "includeRawCopyrights": "0",
                # "teiCoordinates": "p",
                "teiCoordinates": "formula",
                "segmentSentences": "1"
            }

            request_url = GROBID_URL + '/api/processFulltextDocument'

            try:
                response = requests.post(request_url, files=files, data=data)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"❌ Błąd przetwarzania {pdf_full_path}: {e}")
                return

            with open(xml_output_path, "wb") as xml_file:
                xml_file.write(response.content)

