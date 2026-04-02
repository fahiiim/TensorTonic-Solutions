import numpy as np

def euclidean_distance(x, y):
    if len(x) != len(y):
        raise ValueError("okay bhaiya")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(x, y)))