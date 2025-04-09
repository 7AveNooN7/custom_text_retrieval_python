from difflib import SequenceMatcher


def partial_ratio_score(s1, s2):
    """
    Zwraca najwyższe ratio porównania krótszego napisu z
    jakimkolwiek fragmentem dłuższego napisu.
    """
    # Upewniamy się, który jest krótszy
    if len(s1) <= len(s2):
        shorter, longer = s1, s2
    else:
        shorter, longer = s2, s1

    shorter = shorter.lower()
    longer = longer.lower()

    best_score = 0.0
    len_shorter = len(shorter)
    len_longer = len(longer)

    for start in range(len_longer - len_shorter + 1):
        # Pobieramy podciąg dłuższego napisu długości krótszego
        chunk = longer[start:start + len_shorter]
        score = SequenceMatcher(None, chunk, shorter).ratio()
        best_score = max(best_score, score)

        # Możemy dodać wczesne przerwanie pętli, jeśli trafimy 1.0
        if best_score == 1.0:
            break
    #print(f's1: {s1}, s2: {s2} best_score: {best_score}')
    return best_score


def partial_ratio_by_whole_words_score(s1: str, s2: str) -> float:
    """
    Zwraca najwyższe 'ratio' dopasowania listy słów z krótszego napisu
    do jakiejkolwiek podlisty słów w dłuższym napisie.
    Porównuje *całe słowa*, a nie fragmenty znaków.
    """

    # Rozbijamy oba ciągi na listy słów (lowercase)
    words1 = s1.lower().split()
    words2 = s2.lower().split()

    # Jeśli któryś jest pusty, ratio = 0
    if not words1 or not words2:
        return 0.0

    # Ustalamy, który jest krótszy
    if len(words1) <= len(words2):
        shorter_words, longer_words = words1, words2
    else:
        shorter_words, longer_words = words2, words1

    best_score = 0.0
    len_shorter = len(shorter_words)
    len_longer = len(longer_words)

    # Tworzymy wszystkie podlisty w "dłuższym" odpowiadające długości "krótszego"
    # i porównujemy je z "krótszym"
    for start in range(len_longer - len_shorter + 1):
        # Podlista słów
        chunk_words = longer_words[start:start + len_shorter]
        # Sklejamy w ciąg, by użyć SequenceMatcher
        chunk_str = ' '.join(chunk_words)
        shorter_str = ' '.join(shorter_words)

        # Obliczamy ratio
        score = SequenceMatcher(None, chunk_str, shorter_str).ratio()
        if score > best_score:
            best_score = score

        # Jeśli mamy 1.0 (pełne dopasowanie), nie ma co dalej szukać
        if best_score == 1.0:
            break

    return best_score