import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Convert to numpy array
    y = np.asarray(y)
    
    # Validate shape
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    
    # Determine number of classes
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    # Validate labels
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("Labels must be in range [0, num_classes-1]")
    
    # Create one-hot matrix
    N = y.shape[0]
    one_hot_matrix = np.zeros((N, num_classes), dtype=float)
    
    # Vectorized assignment
    one_hot_matrix[np.arange(N), y] = 1.0
    
    return one_hot_matrix