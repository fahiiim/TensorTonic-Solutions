def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    Dict = {}

    for sentence in sentences:
        for word in sentence:
            if word in Dict:
                Dict[word] += 1
            else:
                Dict[word] = 1
    return Dict