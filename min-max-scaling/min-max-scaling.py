import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Convert to numpy array (float for proper division)
    X = np.asarray(data, dtype=float)
    
    # Compute column-wise min and max
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    
    # Compute range
    col_range = col_max - col_min
    
    # Avoid division by zero (constant columns)
    # Replace 0 range with 1 temporarily
    safe_range = np.where(col_range == 0, 1, col_range)
    
    # Apply scaling
    X_scaled = (X - col_min) / safe_range
    
    # Set constant columns explicitly to 0.0
    X_scaled[:, col_range == 0] = 0.0
    
    # Return as list of lists
    return X_scaled.tolist()