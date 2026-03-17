import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # create word---->index mapping
    vocab_index= {word: i for i, word in enumerate(vocab)}

    # initialize zero array
    bow_vector= np.zeros(len(vocab), dtype= int)

    # occurance making 
    for token in tokens:
        if token in vocab_index:
            bow_vector[vocab_index[token]] += 1
            
    return bow_vector