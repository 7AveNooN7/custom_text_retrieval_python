import os
from dataclasses import dataclass
from typing import List, Dict

import pymupdf

from src.pdf_to_txt.string_ratio_scores import partial_ratio_score
from src.pdf_to_txt.utils import clean_title, is_title_similar_to_excluded_titles, \
    is_title_similar_to_excluded_structural_elements, has_numbering


@dataclass
class ChapterInfo:
    title: str
    start_page: int
    end_page: int
    valid: bool

@dataclass
class PdfFileInfo:
    file_path: str
    file_name: str
    chapter_title_repeated_word: str
    chapter_info: Dict[str, ChapterInfo]
    valid: bool


class PdfToTxtAnalysis:
    def __init__(self, path_list: List[str], previous_path_list: List[str]):
        self.path_list = path_list
        self.previous_path_list = previous_path_list

        self.files_paths_to_consider = [item for item in self.path_list if item not in self.previous_path_list]


    def prepare_pdf_information(self):
        pdf_file_info: Dict[str, PdfFileInfo] = {}
        for file_path in self.files_paths_to_consider:
            print(f'iteration ')
            file_name = os.path.basename(file_path)
            doc: pymupdf.Document = pymupdf.open(file_path)
            toc = doc.get_toc()  # Pobiera spis treści

            main_chapters = []

            if not toc:
                print(f"❌ {file_name} - no toc!")
                return "NIE MA TOC"
            else:
                # print(f"{file_name} All entries count: {len(toc)}")
                # print(f"List of entries (Level 1):")
                for entry in toc:
                    if entry[0] == 1:
                        print(f"📌 {entry}")
                        main_chapters.append(entry)

            has_negative_start = any(start_page == -1 for _, _, start_page in main_chapters)
            if not has_negative_start:
                main_chapters.sort(key=lambda x: x[2])


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
                        end_page = start_page_of_next_chapter - 1
                else:
                    end_page = total_pages  # Ostatni rozdział trwa do końca dokumentu

                chapter_info: ChapterInfo = ChapterInfo(
                    title=title,
                    start_page=start_page,
                    end_page=end_page,
                    valid=False
                )
                chapters_info[title] = chapter_info

            chapters_to_remove: List[str] = []
            for chapter_title, chapter_info in chapters_info.items():
                remove = False
                if partial_ratio_score(file_name, chapter_title) >= 0.8:
                    # print(f'❌ {title}: Entry rejected - title similar to document name!')
                    remove = True

                if is_title_similar_to_excluded_titles(chapter_title, 0.95):
                    # print(f'❌ {title}: Entry rejected - title similar to exclusion titles list!')
                    remove = True

                if is_title_similar_to_excluded_structural_elements(chapter_title):
                    # print(f'❌ {title}: Entry rejected - title similar to excluded structural elements list!')
                    remove = True

                if remove:
                    chapters_to_remove.append(chapter_title)

            for chapter_to_remove in chapters_to_remove:
                del chapters_info[chapter_to_remove]

            print('FILTERED CHAPTERS:')
            for chapter in chapters_info:
                print(chapter)


            pdf_file_info[file_name] = PdfFileInfo(
                file_path=file_path,
                file_name=file_name,
                chapter_title_repeated_word='',
                chapter_info=chapters_info,
                valid=False
            )

        print('CHAPTERS:')
        for chapter_title, value in chapters_info.items():
            print(value)
        return pdf_file_info

    # def validate_pdf_files(self, *, pdf_files_info: Dict[str, PdfFileInfo]):
    #     valid_chapters = []
    #     invalid_chapters = []
    #     for file_name, pdf_file_info in pdf_files_info.items():
    #         for chapter_name, chapter_info in pdf_file_info.chapter_info.items():
    #             if chapter_info.start_page == -1 or chapter_info.end_page == -1:
    #                 #invalid_chapters.append(chapter_info)



