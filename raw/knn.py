import numpy as np
import pandas as pd
from collections import Counter
from typing import Union

from sklearn.model_selection import train_test_split
from metrics import confusion_matrix, precision_score, recall_score


class KNN:
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if isinstance(X, pd.DataFrame):
            self.X_train = X.values
        else:
            self.X_train = np.array(X)

        if isinstance(y, pd.Series):
            self.y_train = y.values
        else:
            self.y_train = np.array(y)

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _manhattan_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sum(np.abs(point1 - point2))

    def _cosine_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        dot_product = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # If one vector is zero, cosine similarity is undefined, return max distance

        cosine_similarity = dot_product / (norm1 * norm2)
        # Convert similarity to distance (1 - similarity)
        return 1 - cosine_similarity

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(point1, point2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(point1, point2)
        elif self.distance_metric == 'cosine':
            return self._cosine_distance(point1, point2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _get_neighbors(self, test_point: np.ndarray) -> list:
        distances = []

        # Calculate distance from test_point to all training points
        for i, train_point in enumerate(self.X_train):
            dist = self._calculate_distance(test_point, train_point)
            distances.append((dist, i))  # (distance, index)

        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        return neighbors

    def predict_single(self, test_point: np.ndarray) -> int:
        neighbors = self._get_neighbors(test_point)

        # Get the labels of k nearest neighbors
        neighbor_labels = []
        for _, idx in neighbors:
            neighbor_labels.append(self.y_train[idx])

        # Return the most common label among neighbors
        label_counts = Counter(neighbor_labels)
        predicted_label = label_counts.most_common(1)[0][0]

        return predicted_label

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for test_point in X:
            prediction = self.predict_single(test_point)
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values

        correct_predictions = np.sum(predictions == y)
        total_predictions = len(y)

        return correct_predictions / total_predictions

    def precision(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return precision_score(y, predictions)

    def recall(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        predictions = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return recall_score(y, predictions)

    def get_confusion_matrix(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return confusion_matrix(y, predictions)


if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv("heart.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN(k=5, distance_metric="cosine")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = knn.score(X_test, y_test)
    precision = knn.precision(X_test, y_test)
    recall = knn.recall(X_test, y_test)
    conf_matrix = knn.get_confusion_matrix(X_test, y_test)

    print(f"KNN Model with k={knn.k}, distance metric: {knn.distance_metric}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Predictions: {y_pred}")
    print(f"Actual:      {y_test.values}")

    print(f"\nTesting different values of k:")
    for k in [1, 3, 5, 7, 9]:
        knn_temp = KNN(k=k)
        knn_temp.fit(X_train, y_train)
        acc = knn_temp.score(X_test, y_test)
        prec = knn_temp.precision(X_test, y_test)
        rec = knn_temp.recall(X_test, y_test)
        conf_matrix = knn_temp.get_confusion_matrix(X_test, y_test)
        print(conf_matrix)
        print(f"k={k}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
