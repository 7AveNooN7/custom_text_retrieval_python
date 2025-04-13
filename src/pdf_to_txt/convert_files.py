import concurrent
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from lxml import etree

import requests
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
from pymupdf import pymupdf

from src.config import TXT_FOLDER, GROBID_URL
from src.pdf_to_txt.models.file_settings_model import FileSettingsModel, ConversionMethodEnum
from src.pdf_to_txt.pdf_file_info import PdfFileInfo, ChapterInfo
from src.ui.tabs.create_database_tab import get_waiting_css_with_custom_text, get_css_text


class ConvertFiles:
    def __init__(self, files_state: Dict[str, PdfFileInfo], files_settings_state: Dict[str, FileSettingsModel], database_folder_name: str, grobid_semaphore):
        self.files_state: Dict[str, PdfFileInfo] = files_state
        self.files_settings_state: Dict[str, FileSettingsModel] = files_settings_state
        self.database_folder_name: str = database_folder_name
        self.grobid_semaphore = grobid_semaphore

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
        yield get_waiting_css_with_custom_text(text=f"Processing .pdf files (0/{len(list(self.files_state.keys()))})...")
        if len(list(self.files_state.keys())) == 1:
            for file_name, file_value in self.files_state.items():
                self._convert_single_file((file_name, file_value, self.files_settings_state[file_name], txt_folder_path))

        else:
            # Prepare arguments for each file to be processed
            tasks = [
                (file_name, file_value, self.files_settings_state[file_name], txt_folder_path)
                for file_name, file_value in self.files_state.items()
            ]

            max_workers = min(os.cpu_count(), len(tasks))
            print(f'max_workers: {max_workers}')

            # Use multiprocessing Pool to process files in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(self._convert_single_file, tasks)
                for i, _ in enumerate(results, start=1):
                    yield get_waiting_css_with_custom_text(text=f"Processing .pdf files ({i}/{len(tasks)})...")

        yield get_css_text(text="Done")

    def _convert_single_file(self, args: Tuple[str, PdfFileInfo, FileSettingsModel, str]) -> None:
        print(f'_convert_single_file')
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
        print(f'grobid_method')
        current_file_folder_path = os.path.join(folder_path, Path(file.file_name).stem.replace(" ", "_"))
        os.makedirs(current_file_folder_path)

        # tworzy podzielone .pdf dla grobid do przetowrzenia
        new_pdf_paths = self.create_split_pdf_files(file=file, file_settings=file_settings, current_file_folder_path=current_file_folder_path)

        for new_pdf in new_pdf_paths:
            print(f'{new_pdf}')

        # tworzy xml'e z podzielonych .pdf
        self.process_files_with_grobid(new_pdf_paths=new_pdf_paths)

    def create_split_pdf_files(self, file: PdfFileInfo, file_settings: FileSettingsModel,
                               current_file_folder_path: str) -> List[str]:
        doc: pymupdf.Document = pymupdf.open(file.file_path)

        chapters_to_iterate_for = file.filtered_chapter_info if (
                    file_settings.use_filter and file.filtered_toc) else file.chapter_info
        if chapters_to_iterate_for is None:
            chapters_to_iterate_for = file.synthetic_chapter_info

        def process_chapter(chapter_data: ChapterInfo):
            print(f'chapterInfo: {chapter_data}')
            reserve_name, chapter_info_value = chapter_data
            chapter_info_value = chapter_info_value[1]
            new_doc = pymupdf.open()
            new_doc.insert_pdf(doc, from_page=chapter_info_value.start_page - 1,
                               to_page=chapter_info_value.end_page - 1)

            new_pdf_file_name = self.make_valid_filename(filename=chapter_info_value.title,
                                                         name_replacement=str(reserve_name)) + ".pdf"

            new_pdf_path = os.path.join(current_file_folder_path, new_pdf_file_name)
            new_doc.save(new_pdf_path)
            new_doc.close()
            return new_pdf_path

        with ThreadPoolExecutor() as executor:
            # Mapujemy procesowanie rozdziałów na pulę wątków
            new_pdf_paths = list(executor.map(process_chapter, enumerate(chapters_to_iterate_for.items())))

        doc.close()
        return new_pdf_paths


    def process_files_with_grobid(self, new_pdf_paths: List[str]):
        # lista wszystkich pelnych sciezek plikow .pdf

        # Semaphore blokuje i tak do 8
        with ThreadPoolExecutor() as executor:
            # Tworzymy przyszłe zadania dla każdego pliku PDF
            futures = {
                executor.submit(self.process_with_grobid, pdf): pdf
                for pdf in new_pdf_paths
            }

            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    final_string = future.result()
                    print(f"Otrzymano wynik dla pliku {pdf_path}")
                except Exception as exc:
                    print(f"Błąd dla pliku {pdf_path}: {exc}")

    def process_with_grobid(self, pdf_full_path: str) -> str:
        xml_output_path = pdf_full_path.replace(".pdf", ".xml")
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
                "segmentSentences": "0" # do segment_senteces wymagane jest wykrycie jezyka a sie to wykrzacza
            }

            request_url = GROBID_URL + '/api/processFulltextDocument'

            with self.grobid_semaphore:
                try:
                    response = requests.post(request_url, files=files, data=data)
                    response.raise_for_status()
                except requests.RequestException as e:
                    print(f"❌ Błąd przetwarzania {pdf_full_path}: {e}")
                    return

                with open(xml_output_path, "wb") as xml_file:
                    xml_file.write(response.content)
                    xml_string = response.content
                    final_text = self.parse_grobid_xml_to_txt(xml_string, pdf_full_path)
                    return final_text

    def parse_grobid_xml_to_txt(self, xml_string: str, pdf_full_path: str) -> str:
        """
        Ekstrahuje tekst wyłącznie z <p> wewnątrz <div> w sekcji <body> XML TEI, ignorując inne elementy.
        Łączy kolejne <p> dodając pojedynczą spację między nimi.
        Uwzględnia brak segmentacji zdań (brak znaczników <s>).
        """
        text_output_path = pdf_full_path.replace(".pdf", ".txt")

        parser = etree.XMLParser(remove_blank_text=True)
        try:
            tree = etree.fromstring(xml_string, parser)
        except etree.XMLSyntaxError:
            print("Błąd parsowania XML.")
            return ""

        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

        body = tree.find('.//tei:body', namespaces=namespaces)
        if body is None:
            print("Brak sekcji <body> w podanym XML.")
            return ""

        extracted_text = []

        for div in body.findall('./tei:div', namespaces=namespaces):
            paragraph_texts = []

            for p in div.findall('.//tei:p', namespaces=namespaces):
                # Pobieramy cały tekst z akapitu
                paragraph_text = " ".join(p.itertext()).strip()

                # Sprawdzamy, czy tekst akapitu jest niepusty
                if paragraph_text:
                    # Opcjonalnie: możemy sprawdzić, czy tekst spełnia jakieś kryteria (np. wielka litera na początku)
                    #if re.match(r"^[A-ZĄĆĘŁŃÓŚŹŻ]", paragraph_text):
                    paragraph_texts.append(paragraph_text)

            if paragraph_texts:
                extracted_text.append(" ".join(paragraph_texts))

        extracted_text = "\n".join(extracted_text)
        with open(text_output_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

        return extracted_text

    def make_valid_filename(self, filename: str, name_replacement: str, os_type: str = "windows", replacement: str = "_") -> str:
        """
        Przyjmuje nazwę pliku i zwraca poprawną nazwę zgodną z wymaganiami systemu operacyjnego.
        Usuwa niedozwolone znaki, skraca nazwę, jeśli jest za długa, i unika zarezerwowanych nazw.

        Args:
            filename (str): Nazwa pliku do poprawienia.
            os_type (str): System operacyjny ("windows" lub "posix" dla Linux/macOS).
            replacement (str): Znak zastępujący niedozwolone znaki (domyślnie "_").

        Returns:
            str: Poprawna nazwa pliku.
        """
        if not filename or filename.isspace():
            return "default_filename"

        # Usuwamy białe znaki na początku i końcu
        filename = filename.strip()

        # Zamiana spacji na replacement (np. '_')
        filename = filename.replace(" ", replacement)

        # Maksymalna długość nazwy pliku
        max_length = 255

        if os_type.lower() == "windows":
            # Zarezerwowane nazwy w Windows
            reserved_names = {
                "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3",
                "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            }

            # Sprawdzenie zarezerwowanych nazw
            base_name = filename.split(".")[0].upper()
            if base_name in reserved_names:
                filename = f"_{filename}"

            # Zastępujemy niedozwolone znaki: < > : " / \ | ? *
            invalid_chars = r'[<>:"/\\|?*]'
            filename = re.sub(invalid_chars, replacement, filename)

            # Usuwamy kropki na początku lub końcu
            filename = filename.strip(".")

        else:  # POSIX (Linux, macOS)
            # Zastępujemy niedozwolone znaki: / i null byte
            filename = filename.replace("/", replacement).replace("\0", "")

        # Zastępujemy wielokrotne spacje pojedynczą spacją
        filename = re.sub(r"\s+", " ", filename)

        # Skracamy nazwę, jeśli jest za długa (uwzględniając kodowanie UTF-8)
        encoded = filename.encode('utf-8')
        if len(encoded) > max_length:
            # Zachowujemy rozszerzenie, jeśli istnieje
            base, ext = (filename.rsplit(".", 1) if "." in filename else (filename, ""))
            # Obliczamy maksymalną długość bazowej nazwy
            ext_len = len(ext.encode('utf-8')) + (1 if ext else 0)  # +1 dla kropki
            max_base_len = max_length - ext_len
            # Skracamy nazwę bazową
            while len(base.encode('utf-8')) > max_base_len and base:
                base = base[:-1]
            filename = f"{base}.{ext}" if ext else base

        # Jeśli nazwa nadal jest pusta, zwracamy domyślną
        if not filename:
            return name_replacement

        # Ostateczne sprawdzenie poprawności
        try:
            test = Path(filename).name
            return filename
        except (ValueError, OSError):
            # W razie problemów zwracamy nazwę z dodatkowym prefiksem
            return f"corrected_{filename}"