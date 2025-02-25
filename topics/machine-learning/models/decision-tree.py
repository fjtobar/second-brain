import numpy as np

### Explanation ### 
# Decision Trees are a type of Supervised Machine Learning where the data is continuously split according to a certain parameter.
# The tree can be explained by two entities, namely decision nodes and leaves.
# Is an acyclic graph that can be used to make decisions
# We want to predict the class given a feature vector
# We consider the ID3 algorithm to build the tree (Iterative Dichotomiser 3)
# The optimization criterion is the average log-likelihood
# The decision tree is a non-parametric model, which means that it does not make any assumptions about the underlying data distribution.
# To evaluate how good a split is, we use the entropy: H(S) = -Σ p_i log(p_i), where p_i is the proportion of class i in the set S.
# The algorithm stop at a leaf node when 
  # all examples in the leaf node are classified correctly
  # we cannot find an attribute to split on
  # the split reduces the entropy by less than a certain threshold 
  # the tree reaches a maximum depth

# The algorithm does not garantee the best tree, it is a greedy algorithm.

class DecisionTree:
    """
    A simple Decision Tree model for classification using ID3 algorithm.

    Attributes:
    -----------
    max_depth : int
        The maximum depth of the decision tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    criterion : str
        The criterion to evaluate the quality of a split.
    
    Methods:
    --------
    fit(X, y):
        Trains the decision tree model using the given training data.
    predict(X):
        Predicts the target values for the given input features.
    """

    def __init__(self, max_depth=5, min_samples_split=2, criterion='entropy'):
        """Initialize the decision tree model."""
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.criterion         = criterion
        self.tree              = None

    def _entropy(self, y):
        """Calculate Entropy for classification (ID3 Algorithm)."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Small epsilon to prevent log(0)

    def _find_best_split(self, X, y):
        """Find the best split for the current node."""
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_score = None, None, float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices  = np.where(X[:, feature] < threshold)[0]
                right_indices = np.where(X[:, feature] >= threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue #skip if not enoigh samples

                left_entropy  = self._entropy(y[left_indices])
                right_entropy = self._entropy(y[right_indices])

                score = (len(left_indices) / n_samples) * left_entropy + (len(right_indices) / n_samples) * right_entropy

                if score < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score


        if best_feature is None:
            return None, None

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping conditinos 
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            if len(y) == 0:
                return None
            return np.bincount(y).argmax()

        # find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return np.bincount(y).argmax()

        # split the dataset
        left_mask  = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree model"""
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        """Recursively predict a single sample."""

        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self._predict_sample(x, node['left'])
            else:
                return self._predict_sample(x, node['right'])
        else:
            return node # leaf node

    def predict(self, X):
        """Predict class labels"""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """Compute accuracy for classification"""
        y_pred = self.predict(X)

        return np.mean(y_pred == y)

    def _print_tree(self, node, depth=0):
        """Recursively prints the decision tree structure."""
        if isinstance(node, dict):
            print(f"{'  ' * depth}├── Feature {node['feature']} ≤ {node['threshold']}")
            self._print_tree(node['left'], depth + 1)
            print(f"{'  ' * depth}└── Feature {node['feature']} > {node['threshold']}")
            self._print_tree(node['right'], depth + 1)
        else:
            print(f"{'  ' * depth}🎯 Class: {node}")

    def print_tree(self):
        """Public method to print the tree structure."""
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