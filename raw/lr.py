import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from metrics import confusion_matrix, accuracy_score, precision_score, recall_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32, epochs: int = 1000,
                 random_state: int = 42):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None
        self.bias = 0
        self.scaler = StandardScaler()
        self.cost_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        m = X.shape[0]  # Number of samples
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)

        return dw, db

    def fit(self, X: pd.DataFrame, y: pd.Series):
        np.random.seed(self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.array(X_scaled)
        y = np.array(y)

        n_features = X_scaled.shape[1]
        self.weights = np.random.normal(0, 0.01, size=(n_features,))

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X_scaled))
            X_shuffled = X_scaled[indices]
            y_shuffled = y[indices]
            epoch_cost = 0
            num_batches = 0

            for i in range(0, len(X_shuffled), self.batch_size):

                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(z)

                batch_loss = self._cross_entropy_loss(y_batch, y_pred)
                epoch_cost += batch_loss
                num_batches += 1

                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            avg_cost = epoch_cost / num_batches
            self.cost_history.append(avg_cost)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Cost: {avg_cost:.4f}')

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.array(X_scaled)
        z = np.dot(X_scaled, self.weights) + self.bias
        probabilities = self._sigmoid(z)

        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return accuracy_score(y, predictions)

    def precision(self, X: pd.DataFrame, y: pd.Series) -> float:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return precision_score(y, predictions)

    def recall(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Computes the recall of the model (TP / (TP + FN))

        Parameters:
        - X: Feature dataframe
        - y: True target series

        Returns:
        - Recall score
        """
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return recall_score(y, predictions)

    def _calculate_auc_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_scores_sorted = y_scores[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

        # Convert to boolean array to count true positives and false positives
        tps = np.cumsum(y_true_sorted)[threshold_idxs]
        fps = np.cumsum(1 - y_true_sorted)[threshold_idxs]

        # Add an extra threshold position to start with 0 false positive
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if len(np.unique(y_true)) != 2:
            return 0.5

        fpr = fps / fps[-1]  # False positive rate
        tpr = tps / tps[-1]  # True positive rate

        # Calculate AUC using trapezoidal rule
        return np.trapz(tpr, fpr)

    def auc_roc(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_scores = self.predict_proba(X)
        auc_score = self._calculate_auc_roc(np.array(y), y_scores)
        return auc_score

    def get_confusion_matrix(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        predictions = self.predict(X)

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        return confusion_matrix(y, predictions)


if __name__ == "__main__":
    df = pd.read_csv("heart.csv")

    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression(learning_rate=0.01, batch_size=16, epochs=500)
    model.fit(X_train, y_train)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Calculate confusion matrix
    train_conf_matrix = model.get_confusion_matrix(X_train, y_train)
    test_conf_matrix = model.get_confusion_matrix(X_test, y_test)

    # Calculate additional metrics
    train_precision = model.precision(X_train, y_train)
    test_precision = model.precision(X_test, y_test)

    train_recall = model.recall(X_train, y_train)
    test_recall = model.recall(X_test, y_test)

    train_auc_roc = model.auc_roc(X_train, y_train)
    test_auc_roc = model.auc_roc(X_test, y_test)

    print(f"Training Metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  AUC-ROC: {train_auc_roc:.4f}")
    print("  Confusion Matrix:")
    print(train_conf_matrix)

    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}")
    print("  Confusion Matrix:")
    print(test_conf_matrix)

    # Show cost history
    print(f"\nFinal cost: {model.cost_history[-1]:.4f}")
