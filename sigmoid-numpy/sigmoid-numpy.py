import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype= float)
    s = 1 / ( 1 + np.exp(-x) )# Write code here
    return s