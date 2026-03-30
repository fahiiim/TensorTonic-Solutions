import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.array(X, dtype=float)  # ensure float + copy-safe
    result = X.copy()

    # Handle 1D case
    if X.ndim == 1:
        valid = ~np.isnan(X)

        if np.any(valid):
            if strategy == 'mean':
                fill_value = np.mean(X[valid])
            else:
                fill_value = np.median(X[valid])
        else:
            fill_value = 0.0

        result[~valid] = fill_value
        return result

    # Handle 2D case
    n_rows, n_cols = X.shape

    for j in range(n_cols):
        col = X[:, j]
        valid = ~np.isnan(col)

        if np.any(valid):
            if strategy == 'mean':
                fill_value = np.mean(col[valid])
            else:
                fill_value = np.median(col[valid])
        else:
            fill_value = 0.0

        result[~valid, j] = fill_value

    return result