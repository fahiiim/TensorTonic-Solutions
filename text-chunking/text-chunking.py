def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)

        if len(tokens) <= i + chunk_size:
            break

    return chunks
        