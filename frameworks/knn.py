import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt


def load_and_prepare_data(filepath: str = "heart.csv"):
    """Load and prepare the heart disease dataset."""
    df = pd.read_csv(filepath)

    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()}")

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def evaluate_model(model, X_test, y_test, model_name="KNN"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"{model_name} Test Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


def test_different_k_values(
    X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9, 11, 15]
):
    """Test KNN with different k values."""
    print(f"\n{'='*50}")
    print("Testing Different K Values:")
    print(f"{'='*50}")

    results = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)

        results.append({"k": k, "accuracy": acc, "precision": prec, "recall": rec})

        print(f"\nk={k}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  Confusion Matrix:\n{conf_mat}")

    return results


def plot_k_values_performance(results):
    """Plot performance metrics for different k values."""
    k_values = [r["k"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker="o", label="Accuracy", linewidth=2)
    plt.plot(k_values, precisions, marker="s", label="Precision", linewidth=2)
    plt.plot(k_values, recalls, marker="^", label="Recall", linewidth=2)

    plt.xlabel("K Value", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("KNN Performance vs K Value", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    plt.tight_layout()
    plt.savefig("knn_k_values_performance.png", dpi=300)
    print("\nPlot saved as 'knn_k_values_performance.png'")
    plt.show()


def cross_validation_analysis(X, y, k_values=[1, 3, 5, 7, 9]):
    """Perform cross-validation for different k values."""
    print(f"\n{'='*50}")
    print("Cross-Validation Analysis (5-fold):")
    print(f"{'='*50}")

    # Scale features for cross-validation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy")

        print(f"\nk={k}")
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    X, y = load_and_prepare_data("heart.csv")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Feature scaling (CRITICAL for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling applied (StandardScaler)")

    # Train KNN with optimal k (we'll use k=1 based on your results)
    print(f"\n{'='*50}")
    print("Training KNN with k=1:")
    print(f"{'='*50}")

    knn_best = KNeighborsClassifier(n_neighbors=1)
    knn_best.fit(X_train_scaled, y_train)

    # Evaluate best model
    evaluate_model(knn_best, X_test_scaled, y_test, "KNN (k=1)")

    # Test different k values
    results = test_different_k_values(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        k_values=[1, 3, 5, 7, 9, 11, 15, 20],
    )

    # Plot results
    plot_k_values_performance(results)

    # Cross-validation analysis
    cross_validation_analysis(X, y, k_values=[1, 3, 5, 7, 9])

    # Try different distance metrics
    print(f"\n{'='*50}")
    print("Testing Different Distance Metrics (k=3):")
    print(f"{'='*50}")

    for metric in ["euclidean", "manhattan", "minkowski"]:
        knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"{metric.capitalize()}: {acc:.4f}")


if __name__ == "__main__":
    main()
