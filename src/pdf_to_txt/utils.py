from collections import defaultdict
from difflib import SequenceMatcher
import re
from typing import List, Set

import unicodedata
from tqdm import tqdm

from src.config import EXCLUDED_TITLES, EXCLUDED_STRUCTURAL_ELEMENTS
from src.pdf_to_txt.string_ratio_scores import partial_ratio_by_whole_words_score


def is_title_similar_to_excluded_titles(title, threshold):
    for excluded_title in EXCLUDED_TITLES:
        if partial_ratio_by_whole_words_score(title, excluded_title) > threshold:
            return True
    return False

def is_title_similar_to_excluded_structural_elements(title):
    for excluded_structural_element in EXCLUDED_STRUCTURAL_ELEMENTS:
        split_title = title.split()
        if SequenceMatcher(None, excluded_structural_element, split_title[0]).ratio() > 0.95:
            return True
    return False

def has_numbering(title):
    """Sprawdza, czy tytuł zawiera numerację rzymską lub arabską."""
    return bool(re.search(r'\b(\d+|[IVXLCDM]+)\b', title, re.IGNORECASE))

def clean_title(title):
    # Normalizacja Unicode (usuwa ukryte znaki)
    title = unicodedata.normalize("NFKC", title)

    # Zastąpienie znaków nowej linii (CR, LF) spacją
    title = title.replace('\r', ' ').replace('\n', ' ')

    # Usunięcie niewidocznych lub zerowej szerokości znaków
    # (czyli wszystkich znaków niebędących "drukowalnymi")
    title = ''.join(ch for ch in title if ch.isprintable())

    if ".pdf" in title:
        #print(f'❌ {title}: Entry changed - title contains ".pdf!"')
        title = title.replace(".pdf", "")

    if ":" in title:
        #print(f'❌ {title}: Entry changed - title contains ":"!')
        title = title.replace(":", "")

    # Przycięcie białych znaków na początku i końcu
    return title.strip()

roman_or_arabic_pattern = re.compile(r"^(?:[IVXLCDMivxlcdm]+|\d+)$")

def clean_title_from_repeated_words(title: str, repeated_words: list[str]) -> str:
    """
    Usuwa z tytułu słowa z listy repeated_words
    wraz z następującym po nich tokenem,
    jeśli ten token wygląda na liczbę arabską lub rzymską.

    Przykład:
      "Chapter 1 Blood sampling" -> "Blood sampling"
      (jeśli "chapter" jest w repeated_words)
    """
    # Rozbijamy tytuł na słowa
    tokens = title.split()

    cleaned_tokens = []
    skip_next = False  # flaga informująca, że następny token też omijamy

    for i in range(len(tokens)):
        if skip_next:
            # poprzedni token był "chapter" i kolejny jest liczbą -> pomijamy TEN token
            skip_next = False
            continue

        current_token = tokens[i]

        # Sprawdzamy, czy to "repeated word" (np. "chapter")
        if current_token.lower() in repeated_words:
            # Sprawdzamy, czy kolejny token istnieje i wygląda na liczbę arabską lub rzymską
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if roman_or_arabic_pattern.match(next_token):
                    # jeśli tak, to ustawiamy skip_next, aby pominąć go w kolejnej iteracji
                    skip_next = True
            # i tak nie dodajemy bieżącego tokena do wyniku
        else:
            # Normalny token, który nie jest "repeated word"
            cleaned_tokens.append(current_token)

    # Sklejamy z powrotem w string
    return " ".join(cleaned_tokens)


def are_titles_similar(searched_title: str, text_from_block: str, repeated_words: list[str], threshold=0.7):
    """
    Porównuje dwa tytuły i zwraca True, jeśli są wystarczająco podobne.
    - Jeśli są identyczne → True
    - Jeśli oba mają numerację (np. 'PART I' i 'PART II'), wymagamy precyzyjnego dopasowania
    - Jeśli są tylko częściowo podobne, ale różnią się istotnie, zwracamy False
    """

    searched_title = clean_title(searched_title)
    searched_title = clean_title_from_repeated_words(searched_title, repeated_words)
    searched_title = searched_title.strip().lower()

    text_from_block = clean_title(text_from_block)
    text_from_block = clean_title_from_repeated_words(title=text_from_block, repeated_words=repeated_words)
    text_from_block = text_from_block.strip().lower()

    # 3️⃣ Klasyczne podobieństwo, jeśli nie mają numeracji
    similarity = SequenceMatcher(None, searched_title, text_from_block).ratio()

    return similarity >= threshold  # Standardowy próg podobieństwa (np. 90%)


def find_missing_chapters(doc, chapters_info, repeated):
    """
    Jedna lista: chapters_info = valid + invalid
    Tyle że 'valid' już mają start_page i end_page ustawione,
    'invalid' mają -1.
    """
    total_pages = len(doc)

    i = 0
    n = len(chapters_info)

    progress_bar = tqdm(range(total_pages), desc="📖 Przeszukiwanie stron PDF", unit="strona", leave=True,
                        dynamic_ncols=True)

    for page_num in range(total_pages):
        if i >= n:
            break

        # tqdm.write(f"🔍 Strona {page_num + 1} | Sprawdzam rozdział: {chapters_info[i]['title'] if i < n else 'Koniec'}")
        # 🔹 Aktualizujemy pasek tqdm, żeby pokazywał aktualny rozdział
        progress_bar.set_postfix(
            {"Strona": page_num + 1, "Szukany rozdział": chapters_info[i]["title"] if i < n else "Koniec"})
        progress_bar.update(1)

        page = doc[page_num]
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: b[1])
        top_blocks = blocks_sorted[:3]

        while i < n:
            current_ch = chapters_info[i]

            # Czy ten rozdział jest "invalid" (nieustalony)?
            if current_ch["start_page"] != -1:
                # On jest "valid", więc idziemy do następnego
                i += 1
                continue

            searched_title = current_ch["title"]

            # Sprawdzamy czy w top-5 blokach jest 'searched_title'
            repeated_word = repeated[0]

            searched_title_found = False
            title_matched_block_text = ''

            key_word_found = False
            excluded_found = False
            for block in top_blocks:
                block_text = block[4]

                if searched_title_found == False and searched_title and are_titles_similar(
                        searched_title=searched_title, text_from_block=block_text, repeated_words=repeated,
                        threshold=0.85):
                    title_matched_block_text = block_text
                    searched_title_found = True

                if excluded_found == False and searched_title_found and is_title_similar_to_excluded_titles(
                        block_text, 0.75):
                    for tb in top_blocks:
                        # tb[4] to tekst w danym bloku
                        if len(tb[4]) < 100 and is_title_similar_to_excluded_titles(tb[4], 0.75):
                            tqdm.write(
                                f"❌ '{searched_title}' — pominięty, ponieważ w top-blocks znaleziono EXCLUSION_TITLES")
                            excluded_found = True
                            break

                if key_word_found == False and partial_ratio_by_whole_words_score(block_text, repeated_word) > 0.8:
                    key_word_found = True

            if searched_title_found and key_word_found and not excluded_found:
                # Zwiększamy i, żeby przejść do kolejnego rozdziału
                current_ch["start_page"] = page_num
                tqdm.write(f"✅ {searched_title} -> start_page = {page_num}")
                i += 1
                # PRINT
                tqdm.write(f'Found against block:\n{title_matched_block_text.strip()}')
            else:
                # Nie znaleźliśmy nic na tej stronie -> idziemy do next strony
                break

    # -> (C) Po zakończeniu pętli stronic => ustalamy end_page
    #    W oparciu o (i+1).start_page lub total_pages

    for idx in range(n):
        # Jeśli end_page jest -1
        if chapters_info[idx]["end_page"] == -1:
            if idx + 1 < n:
                # Jeśli rozdział i+1 ma start_page != -1
                next_sp = chapters_info[idx + 1]["start_page"]
                if next_sp != -1:
                    chapters_info[idx]["end_page"] = (next_sp - 1)
                    tqdm.write(f"{chapters_info[idx]['title']} -> end_page={next_sp - 1}")
                else:
                    # ten też jest -1 => brak pewności, zostawimy do końca PDF
                    chapters_info[idx]["end_page"] = total_pages
                    tqdm.write(f"{chapters_info[idx]['title']} -> end_page={total_pages}")
            else:
                # ostatni rozdział
                chapters_info[idx]["end_page"] = total_pages
                tqdm.write(f"{chapters_info[idx]['title']} -> end_page={total_pages}")

    # -> (D) Zwróć chapters_info (wszystkie rozdziały)
    return chapters_info


def get_repeated_words(titles_words_sets: List[Set[str]], threshold: int = 2) -> list:
    """
    Zwraca listę słów, które występują najczęściej w tytułach, pod warunkiem,
    że słowo pojawiło się w co najmniej `threshold` różnych tytułach.

    :param titles_words_sets: lista zbiorów słów, np. [ {"chapter", "1"}, {"chapter", "2"}, ... ]
    :param threshold: minimalna liczba tytułów, w których słowo musi się pojawić,
                      aby w ogóle wziąć je pod uwagę
    :return: lista słów (1 lub więcej, jeśli jest remis), które występują najczęściej,
             spełniając próg `threshold`
    """
    # 1. Liczymy liczbę tytułów, w których występuje dane słowo
    word_in_titles_count = defaultdict(int)
    for words_set in titles_words_sets:
        for word in words_set:
            word_in_titles_count[word] += 1

    # 2. Które słowa przeszły próg `threshold`?
    #    (czyli występują w >= threshold tytułach)
    candidates = [(word, count) for word, count in word_in_titles_count.items()
                  if count >= threshold]

    # Jeśli żadne słowo nie przekroczyło progu, zwracamy pustą listę
    if not candidates:
        return []

    # 3. Spośród słów, które przeszły próg, wybieramy to (lub te), które występują najczęściej
    max_count = max(count for _, count in candidates)  # najwyższa liczba tytułów
    most_frequent_words = [word for word, count in candidates if count == max_count]

    return most_frequent_words