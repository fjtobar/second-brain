import numpy as np

############# Explanation ##############

# A Decision Tree is a supervised learning model used for both classification and regression problems.
# The tree can be explained by two entities, namely decision nodes and leaves.
# It is an acyclic graph that can be used to make decisions.
# We want to predict the class given a feature vector.
# The algorithm used is the ID3 algorithm (Iterative Dichotomiser 3).
# The optimization criterion is the average log-likelihood.
# Decision Trees are non-parametric models, meaning they do not make any assumptions about the underlying data distribution.
# 
# To evaluate how good a split is, we use entropy: H(S) = -Σ p_i log(p_i), where p_i is the proportion of class i in set S.
# 
# The algorithm stops at a leaf node when:
#  - All examples in the leaf node are classified correctly.
#  - No attribute can be used to split further.
#  - The entropy reduction from the split is below a certain threshold.
#  - The tree reaches the maximum depth.
# 
# The ID3 algorithm does not guarantee the best tree, as it is a greedy algorithm.
########################################


class DecisionTree:
    """
    A Decision Tree model for classification using the ID3 algorithm.
    Supports stopping conditions and entropy-based splitting.

    Attributes:
    -----------
    max_depth : int
        The maximum depth of the decision tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    criterion : str
        The criterion used to measure the quality of a split ("entropy").
    feature_names : list, optional
        List of feature names for interpretability.

    Methods:
    --------
    fit(X, y):
        Trains the decision tree model using the given training data.
    predict(X):
        Predicts the target values for the given input features.
    score(X, y):
        Computes the accuracy of the model on a given dataset.
    print_tree():
        Prints the structure of the trained decision tree.
    """

    def __init__(self, max_depth=5, min_samples_split=2, criterion='entropy', feature_names=None):
        """
        Initialize the decision tree model.

        Parameters:
        -----------
        max_depth : int, optional
            The maximum depth of the tree (default is 5).
        min_samples_split : int, optional
            The minimum number of samples required to split a node (default is 2).
        criterion : str, optional
            The criterion to measure the quality of a split (default is "entropy").
        feature_names : list, optional
            List of feature names for better interpretability.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.feature_names = feature_names

    def _entropy(self, y):
        """
        Calculate entropy for classification (ID3 Algorithm).

        Parameters:
        -----------
        y : numpy.ndarray
            The target values of shape (n_samples,).

        Returns:
        --------
        float
            The entropy value.
        """
        if len(y) == 0:
            return 0  # No entropy if there are no samples

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Small epsilon to prevent log(0)

    def _find_best_split(self, X, y, min_entropy_decrease=0.01):
        """
        Find the best feature and threshold to split on, with a minimum entropy gain threshold.

        Parameters:
        -----------
        X : numpy.ndarray
            The input feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).
        min_entropy_decrease : float, optional
            The minimum entropy reduction required to consider a split (default is 0.01).

        Returns:
        --------
        best_feature : int
            The index of the best feature to split on.
        best_threshold : float
            The threshold value for the best split.
        """
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_score = None, None, float("inf")
        current_entropy = self._entropy(y)

        for feature in range(n_features):
            unique_thresholds = np.unique(X[:, feature])

            for threshold in unique_thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # Skip invalid splits

                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                weighted_entropy = (np.sum(left_mask) / n_samples) * left_entropy + \
                                   (np.sum(right_mask) / n_samples) * right_entropy

                entropy_decrease = current_entropy - weighted_entropy
                if entropy_decrease < min_entropy_decrease:
                    continue  # Skip weak splits

                if weighted_entropy < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, weighted_entropy

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        -----------
        X : numpy.ndarray
            The input feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).
        depth : int, optional
            The current depth of the tree (default is 0).

        Returns:
        --------
        dict or int
            A dictionary representing the decision node or an integer representing a leaf class.
        """
        n_samples = X.shape[0]

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y)) if len(y) > 0 else 0  # Default to majority class

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.argmax(np.bincount(y))

        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.argmax(np.bincount(y))

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree model."""
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        """
        Recursively predict a single sample.

        Parameters:
        -----------
        x : numpy.ndarray
            The input feature array of shape (n_features,).
        node : dict or int
            The current node of the tree.

        Returns:
        --------
        int
            The predicted class label.
        """
        if isinstance(node, dict):
            if x[node['feature']] < node['threshold']:
                return self._predict_sample(x, node['left'])
            else:
                return self._predict_sample(x, node['right'])
        return node

    def predict(self, X):
        """Predict class labels for input data."""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """Compute accuracy for classification."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _print_tree(self, node, depth=0):
        """Recursively prints the decision tree structure with feature names."""
        if isinstance(node, dict):
            feature_label = self.feature_names[node['feature']] if self.feature_names else f"Feature {node['feature']}"
            print(f"{'  ' * depth}├── {feature_label} ≤ {node['threshold']:.3f}")
            self._print_tree(node['left'], depth + 1)
            print(f"{'  ' * depth}└── {feature_label} > {node['threshold']:.3f}")
            self._print_tree(node['right'], depth + 1)
        else:
            print(f"{'  ' * depth}🎯 Class: {node}")

    def print_tree(self):
        """Print the decision tree structure."""
        if self.tree is None:
            print("Decision Tree has not been trained yet.")
        else:
            self._print_tree(self.tree)


# Example Usage
if __name__ == "__main__":
    # Sample data for classification
    X_classification = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]])
    y_classification = np.array([0, 0, 0, 1, 1, 1])

    # Decision Tree Classifier (ID3 using entropy)
    clf = DecisionTree(max_depth=3)
    clf.fit(X_classification, y_classification)

    print(f"Classification Accuracy: {clf.score(X_classification, y_classification):.2f}")
    clf.print_tree()
