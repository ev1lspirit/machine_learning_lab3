from functools import cached_property
from scipy.stats import entropy
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


class ContiniousFeatureEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_midpoints(self, values: np.ndarray):
        midpoints = []
        index = 1
        while index < len(values):
            midpoint = (values[index - 1] + values[index]) / 2
            midpoints.append(midpoint)
            index += 1
        return midpoints

    def _split_feature_by_midpoint(self, vector: pd.Series, midpoint: float):
        higher= vector[vector > midpoint]
        lower = vector[vector <= midpoint]
        return higher, lower

    def _count_entropy(self, vector: pd.Series):
        return entropy(vector.value_counts(normalize=True))

    def split_into_classes(self, feature_name: str):
        sorted_feature = self.df[feature_name].sort_values()
        unique_values = sorted_feature.unique()
        midpoints = self._get_midpoints(unique_values)
        total_entropy = self._count_entropy(sorted_feature)
        gains = {}

        for midpoint in midpoints:
            higher, lower = self._split_feature_by_midpoint(sorted_feature, midpoint)
            higher_entropy = self._count_entropy(higher)
            lower_entropy = self._count_entropy(lower)
            midpoint_gain = total_entropy - (
                len(lower) / len(sorted_feature) * lower_entropy
            ) - (len(higher) / len(sorted_feature) * higher_entropy)
            gains[midpoint] = midpoint_gain
        best_midpoint, _= max(gains.items(), key=lambda pair: pair[1])
        return self._split_feature_by_midpoint(sorted_feature, best_midpoint)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Feature to split on (for internal nodes)
        self.threshold = threshold      # Threshold to split on (for continuous features)
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Predicted value (for leaf nodes)


class DecisionTree:

    def __init__(self, dataset, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.dataset = dataset.copy()
        self.target = self.dataset.pop("target")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset, self.target, test_size=0.3, random_state=42
        )
        standart_scaler = StandardScaler()
        self.continuous_encoder = ContiniousFeatureEncoder(self.dataset)
        self.X_train_scaled = standart_scaler.fit_transform(self.X_train)
        self.X_test_scaled = standart_scaler.transform(self.X_test)

        # Parameters for tree building
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None


    @cached_property
    def probabilities_for_each_class(self):
        return self.target.value_counts(normalize=True)

    @cached_property
    def dataset_entropy(self):
        probabilities = self.probabilities_for_each_class
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_entropy(self, y):
        """Calculate entropy of a series of labels"""
        if len(y) == 0:
            return 0
        counts = y.value_counts()
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to prevent log(0)

    def _calculate_information_gain_discrete(self, X, y, feature):
        """Calculate information gain for discrete features"""
        total_entropy = self._calculate_entropy(y)

        # Get unique values of the feature
        feature_values = X[feature].unique()

        # Calculate weighted entropy after split
        weighted_entropy = 0
        for value in feature_values:
            subset = X[X[feature] == value]
            subset_y = y.loc[subset.index]
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * self._calculate_entropy(subset_y)

        information_gain = total_entropy - weighted_entropy
        return information_gain

    def _calculate_information_gain_continuous(self, X, y, feature):
        """Calculate information gain for continuous features using the encoder"""
        sorted_feature = X[feature].sort_values()
        unique_values = sorted_feature.unique()

        if len(unique_values) < 2:
            return 0, None  # Not enough values to split

        midpoints = self.continuous_encoder._get_midpoints(unique_values)
        total_entropy = self._calculate_entropy(y)
        best_gain = -1
        best_threshold = None

        for midpoint in midpoints:
            # Split the data based on the threshold
            higher_mask = X[feature] > midpoint
            lower_mask = X[feature] <= midpoint

            lower_y = y[lower_mask]
            higher_y = y[higher_mask]

            # Calculate weighted entropy after split
            lower_weight = len(lower_y) / len(y)
            higher_weight = len(higher_y) / len(y)

            weighted_entropy = (lower_weight * self._calculate_entropy(lower_y) +
                              higher_weight * self._calculate_entropy(higher_y))

            gain = total_entropy - weighted_entropy

            if gain > best_gain:
                best_gain = gain
                best_threshold = midpoint

        return best_gain, best_threshold

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None  # For continuous features
        best_value = None     # For discrete features

        for feature in X.columns:
            # Check if the feature is continuous by checking its data type
            if pd.api.types.is_numeric_dtype(X[feature]) and len(X[feature].unique()) > 10:  # Assuming numeric with many unique values is continuous
                gain, threshold = self._calculate_information_gain_continuous(X, y, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_value = None
            else:  # Discrete/categorical feature
                gain = self._calculate_information_gain_discrete(X, y, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = None
                    best_value = None  # For discrete splitting we'd need different logic

        return best_feature, best_gain, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree using ID3 algorithm"""
        n_samples, n_features = X.shape
        n_labels = len(y.unique())

        # Stopping criteria
        if (n_labels == 1 or
            depth == self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < self.min_samples_leaf * 2):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_gain, best_threshold = self._find_best_split(X, y)

        # If no good split found, return leaf with most common label
        if best_feature is None or best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create node with best feature
        if best_threshold is not None:  # Continuous feature
            # Split the dataset
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold

            # Create subsets
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]

            # Check if split creates too small subsets
            if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value)

            # Create left and right subtrees
            left_subtree = self._build_tree(X_left, y_left, depth + 1)
            right_subtree = self._build_tree(X_right, y_right, depth + 1)

            return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
        else:  # Discrete feature
            # This would require a more complex implementation for discrete splits
            # For simplicity, using a basic approach
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

    def _most_common_label(self, y):
        """Return the most common label in y"""
        return y.value_counts().index[0]

    def fit(self):
        """Build the decision tree using the training data"""
        self.tree = self._build_tree(self.X_train, self.y_train)
        return self

    def _predict_single(self, x, tree_node):
        """Predict a single sample"""
        if tree_node.value is not None:  # Leaf node
            return tree_node.value

        if x[tree_node.feature] <= tree_node.threshold:
            return self._predict_single(x, tree_node.left)
        else:
            return self._predict_single(x, tree_node.right)

    def predict(self, X):
        predictions = []
        for idx, row in X.iterrows():
            prediction = self._predict_single(row, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

    def evaluate(self):
        """Evaluate the model on test data"""
        y_pred = self.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)
        return accuracy


if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("heart.csv")
    # tree = DecisionTree(dataset=df)
    # tree.fit()
    # accuracy = tree.evaluate()
    # print(f"Model accuracy: {accuracy}")

    # Test with a sample dataset
    import pandas as pd
    import numpy as np

    # Create a simple sample dataset for testing
    data = {
        'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
        'sex': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        'cp': [3, 2, 1, 0, 1, 2, 1, 0, 1, 2],
        'thalach': [150, 108, 123, 130, 172, 122, 150, 152, 174, 140],
        'exang': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        'target': [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Create and train the decision tree
    tree = DecisionTree(dataset=df.copy(), max_depth=3)
    tree.fit()

    # Evaluate the model
    accuracy = tree.evaluate()
    print(f"Model accuracy: {accuracy}")

    # Test predictions on a few samples
    sample_predictions = tree.predict(tree.X_test.head())
    print(f"Sample predictions: {sample_predictions}")
    print(f"Actual values: {tree.y_test.head().values}")
