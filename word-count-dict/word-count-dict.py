def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    Dict = {}

    for sentence in sentences:
        for word in sentence:
            Dict[word] = Dict.get(word, 0) + 1

    return Dict