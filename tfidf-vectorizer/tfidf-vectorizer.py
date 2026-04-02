import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    # Handle empty corpus
    if not documents:
        return np.array([]), []

    # Step 1: Tokenization
    tokenized_docs = []
    for doc in documents:
        if doc:
            tokens = doc.lower().split()
        else:
            tokens = []
        tokenized_docs.append(tokens)

    # Step 2: Build vocabulary (sorted)
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    n_docs = len(documents)
    n_vocab = len(vocab)

    # Handle case with no words at all
    if n_vocab == 0:
        return np.zeros((n_docs, 0)), []

    # Step 3: Document Frequency (df)
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1

    # Step 4: Compute IDF
    idf = {}
    for word in vocab:
        idf[word] = math.log(n_docs / df[word]) if df[word] > 0 else 0.0

    # Step 5: Initialize TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab))

    # Step 6: Compute TF and TF-IDF
    for i, doc in enumerate(tokenized_docs):
        if not doc:
            continue

        term_counts = Counter(doc)
        total_terms = len(doc)

        for word, count in term_counts.items():
            j = vocab_index[word]
            tf = count / total_terms
            tfidf_matrix[i, j] = tf * idf[word]

    return tfidf_matrix, vocab