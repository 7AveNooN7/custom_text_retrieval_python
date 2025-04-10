import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional

import pymupdf

from src.pdf_to_txt.string_ratio_scores import partial_ratio_score
from src.pdf_to_txt.utils import clean_title, is_title_similar_to_excluded_titles, \
    is_title_similar_to_excluded_structural_elements


@dataclass
class ChapterInfo:
    title: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    toc: Optional[bool] = None

@dataclass
class PdfFileInfo:
    file_path: str
    file_name: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    filtered_start_page: Optional[int] = None
    filtered_end_page: Optional[int] = None
    chapter_title_repeated_word: Optional[str] = None
    chapter_info: Optional[Dict[str, ChapterInfo]] = None
    filtered_chapter_info: Optional[Dict[str, ChapterInfo]] = None
    toc: Optional[bool] = None
    filtered_toc: Optional[bool] = None


class PdfToTxtAnalysis:
    def __init__(self, path_list: List[str]):
        self.path_list = path_list

    async def prepare_pdf_information_async(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.prepare_pdf_information)

    def prepare_pdf_information(self):
        pdf_files_info: Dict[str, PdfFileInfo] = {}

        # Zamiast ThreadPoolExecutor –> ProcessPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_single_pdf, self.path_list)

        for result in results:
            pdf_files_info[result.file_name] = result

        return pdf_files_info


    def process_single_pdf(self, file_path: str):
        #print(f'iteration ')
        file_name = os.path.basename(file_path)
        doc: pymupdf.Document = pymupdf.open(file_path)
        toc = doc.get_toc()  # Pobiera spis treści
        #text = "\n\n".join(page.get_text().strip() for page in doc)
        #print(text)
        #print(f'[0]: {doc[1].get_text()}')
        main_chapters = []
        if not toc:
            #print(f"❌ {file_name} - no toc!")
            return PdfFileInfo(
                file_path=file_path,
                file_name=file_name,
                start_page=0,
                end_page=len(doc)
            )
        else:
            for entry in toc:
                if entry[0] == 1:
                    #print(f"📌 {entry}")
                    main_chapters.append(entry)

            total_pages = len(doc)
            chapters_info: Dict[str, ChapterInfo] = {}

            for i, (level, title, start_page) in enumerate(main_chapters):
                title = clean_title(title)
                end_page = None
                if start_page == -1:
                    start_page = None

                if i + 1 < len(main_chapters):
                    start_page_of_next_chapter = main_chapters[i + 1][2]

                    end_page = main_chapters[i + 1][2]

                    if start_page != start_page_of_next_chapter:
                        if end_page == -1:
                            end_page = None
                        else:
                            end_page = start_page_of_next_chapter - 1
                else:
                    end_page = total_pages  # Ostatni rozdział trwa do końca dokumentu

                chapter_info: ChapterInfo = ChapterInfo(
                    title=title,
                    start_page=start_page,
                    end_page=end_page,
                    toc=False
                )
                chapters_info[title] = chapter_info


            filtered_chapters_info: Dict[str, ChapterInfo] = {}

            for chapter_title, chapter_info in chapters_info.items():

                if partial_ratio_score(file_name, chapter_title) >= 0.8:
                    # print(f'❌ {title}: Entry rejected - title similar to document name!')
                    continue

                if is_title_similar_to_excluded_titles(chapter_title, 0.95):
                    # print(f'❌ {title}: Entry rejected - title similar to exclusion titles list!')
                    continue

                if is_title_similar_to_excluded_structural_elements(chapter_title):
                    # print(f'❌ {title}: Entry rejected - title similar to excluded structural elements list!')
                    continue

                filtered_chapters_info[chapter_title] = chapter_info

            return self.validate_pdf_files(
                pdf_file_info=PdfFileInfo(
                    file_path=file_path,
                    file_name=file_name,
                    chapter_title_repeated_word='',
                    chapter_info=chapters_info,
                    filtered_chapter_info=filtered_chapters_info,
                    toc=False,
                    start_page=0,
                    end_page=len(doc)
                )
            )


    def validate_pdf_files(self, *, pdf_file_info: PdfFileInfo) -> PdfFileInfo:
        if pdf_file_info.toc is not None:  # tristate
            # Walidujemy rozdziały w pliku PDF
            updated_chapters = self.validate_chapters(chapters=pdf_file_info.chapter_info)
            pdf_file_info.chapter_info = updated_chapters

            # Ustawiamy pole valid w PdfFileInfo na True, jeśli wszystkie rozdziały są valid
            all_chapters_valid = all(chapter.toc for chapter in pdf_file_info.chapter_info.values())
            pdf_file_info.toc = all_chapters_valid

            updated_filtered_chapters = self.validate_chapters(chapters=pdf_file_info.filtered_chapter_info)
            pdf_file_info.filtered_chapter_info = updated_filtered_chapters

            # Ustawiamy pole valid w PdfFileInfo na True, jeśli wszystkie rozdziały są valid
            all_filtered_chapters_valid = all(chapter.toc for chapter in pdf_file_info.filtered_chapter_info.values())
            pdf_file_info.filtered_toc = all_filtered_chapters_valid

            if pdf_file_info.filtered_toc:
                pdf_file_info.filtered_start_page = list(pdf_file_info.filtered_chapter_info.values())[0].start_page
                pdf_file_info.filtered_start_page = list(pdf_file_info.filtered_chapter_info.values())[-1].end_page


        return pdf_file_info

    def validate_chapters(self, chapters: Dict[str, ChapterInfo]) -> Dict[str, ChapterInfo]:
        chapters_list: List[ChapterInfo] = list(chapters.values())
        if not chapters_list:
            #print("Brak rozdziałów do sprawdzenia.")
            return chapters

        # Rozdziały z pełnymi danymi
        valid_chapters = [ch for ch in chapters_list if ch.start_page is not None and ch.end_page is not None]

        # 1. Wstępna walidacja i zbieranie problemów
        issues = {}
        for chapter in chapters_list:
            chapter.toc = True  # Zakładamy, że jest valid, dopóki nie znajdziemy problemu
            if chapter.start_page is None or chapter.end_page is None:
                chapter.toc = False
                issues[
                    chapter.title] = f"Strony niekompletne (start_page={chapter.start_page}, end_page={chapter.end_page})"
            elif chapter.start_page > chapter.end_page:
                chapter.toc = False
                issues[
                    chapter.title] = f"start_page ({chapter.start_page}) jest większe niż end_page ({chapter.end_page})"
            elif chapter.start_page < 1:
                chapter.toc = False
                issues[chapter.title] = f"start_page ({chapter.start_page}) jest mniejsze niż 1"

        # 2. Sprawdzanie niedozwolonego nakładania się
        for i, chapter in enumerate(valid_chapters):
            for j, other_chapter in enumerate(valid_chapters):
                if i < j and chapter.toc and other_chapter.toc:
                    if not (chapter.end_page < other_chapter.start_page or chapter.start_page > other_chapter.end_page):
                        if not (
                                chapter.end_page == other_chapter.start_page or other_chapter.end_page == chapter.start_page):
                            chapter.toc = False
                            other_chapter.toc = False
                            issues[chapter.title] = issues.get(chapter.title,
                                                               "") + f" Niedozwolone nakładanie się z '{other_chapter.title}' ({other_chapter.start_page}-{other_chapter.end_page})"
                            issues[other_chapter.title] = issues.get(other_chapter.title,
                                                                     "") + f" Niedozwolone nakładanie się z '{chapter.title}' ({chapter.start_page}-{chapter.end_page})"

        # # 3. Raportowanie wyników
        # print("Wyniki walidacji:")
        # for chapter in chapters_list:
        #     if chapter.toc:
        #         print(f"Rozdział '{chapter.title}' ({chapter.start_page}-{chapter.end_page}) jest VALID")
        #     else:
        #         print(
        #             f"Rozdział '{chapter.title}' ({chapter.start_page}-{chapter.end_page}) jest INVALID: {issues.get(chapter.title, 'Nieokreślony problem')}")
        #
        # #4. Podsumowanie
        # print(f"\nPodsumowanie:")
        # print(f"Liczba rozdziałów: {len(chapters_list)}")
        # print(f"Rozdziały z pełnymi danymi: {len(valid_chapters)}")
        # valid_count = sum(1 for ch in chapters_list if ch.toc)
        # print(f"Rozdziały valid: {valid_count}")
        # print(f"Rozdziały invalid: {len(chapters_list) - valid_count}")
        # if valid_chapters:
        #     min_page = min(chapter.start_page for chapter in valid_chapters)
        #     max_page = max(chapter.end_page for chapter in valid_chapters)
        #     print(f"Minimalna strona: {min_page}")
        #     print(f"Maksymalna strona: {max_page}")
        #     print(f"Całkowita liczba stron (dla pełnych danych): {max_page - min_page + 1}")
        # else:
        #     print("Brak rozdziałów z pełnymi danymi do analizy.")
        # print("Walidacja zakończona.")

        # 5. Zwracamy słownik z zaktualizowanymi wartościami valid
        return chapters

    # Wywołanie funkcji