import numpy as np
import pandas as pd
from typing import Union, Tuple


def confusion_matrix(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Get unique labels
    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Create mapping for binary classification
    if len(labels) <= 2:
        # For binary classification, ensure we have 0 and 1
        labels = np.array([0, 1]) if set(labels) <= {0, 1} else np.sort(labels)
    else:
        labels = np.sort(labels)

    # Initialize confusion matrix
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Map actual values to indices
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Fill confusion matrix
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        matrix[true_idx, pred_idx] += 1

    return matrix


def accuracy_score(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    return np.mean(y_true == y_pred)


def precision_score(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calculate True Positives and False Positives
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))

    # Calculate precision
    if (true_positives + false_positives) == 0:
        return 0.0  # Avoid division by zero
    precision_score = true_positives / (true_positives + false_positives)
    return precision_score


def recall_score(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute the recall score (TP / (TP + FN))

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Recall score
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calculate True Positives and False Negatives
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate recall
    if (true_positives + false_negatives) == 0:
        return 0.0  # Avoid division by zero
    recall_score = true_positives / (true_positives + false_negatives)
    return recall_score


def classification_report(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> dict:
    """
    Build a text report showing the main classification metrics

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Dictionary with accuracy, precision, recall, and confusion matrix
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }


def show_classification_report(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]):
    """
    Print a formatted classification report

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    """
    report = classification_report(y_true, y_pred)

    print("Classification Report:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print("Confusion Matrix:")
    print(report['confusion_matrix'])
