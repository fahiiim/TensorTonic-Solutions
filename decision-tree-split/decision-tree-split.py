import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    Returns [feature_index, threshold]
    """
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape

    # Gini impurity function
    def gini(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1 - np.sum(probs ** 2)

    parent_gini = gini(y)

    best_gain = -1
    best_feature = None
    best_threshold = None

    # Iterate over each feature
    for f in range(n_features):
        # Get sorted unique values
        values = sorted(set(X[:, f]))

        # Try midpoints
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            # Split
            left_mask = X[:, f] <= threshold
            right_mask = X[:, f] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            # Compute weighted Gini
            gini_left = gini(y_left)
            gini_right = gini(y_right)

            weighted_gini = (
                (len(y_left) / n_samples) * gini_left +
                (len(y_right) / n_samples) * gini_right
            )

            # Information gain
            gain = parent_gini - weighted_gini

            # Update best split (tie-breaking included)
            if (gain > best_gain or
                (gain == best_gain and (f < best_feature or
                (f == best_feature and threshold < best_threshold)))):
                
                best_gain = gain
                best_feature = f
                best_threshold = threshold

    return [best_feature, best_threshold]