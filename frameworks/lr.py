import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def load_and_prepare_data(filepath: str = "heart.csv"):
    """Load and prepare the heart disease dataset."""
    df = pd.read_csv(filepath)

    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nFeature statistics:")
    print(df.describe())

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name="Test"):
    """Evaluate model and print metrics."""
    if dataset_name == "Training":
        y_pred = model.predict(X_train)
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        y_true = y_train
        X_data = X_train
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_true = y_test
        X_data = X_test

    print(f"\n{'='*50}")
    print(f"{dataset_name} Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return y_pred_proba


def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Logistic Regression", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logistic_regression_roc_curve.png", dpi=300)
    print("\nROC curve saved as 'logistic_regression_roc_curve.png'")
    plt.show()


def plot_feature_importance(model, feature_names):
    """Plot feature coefficients (importance)."""
    coefficients = model.coef_[0]

    # Create dataframe for better visualization
    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Abs_Coefficient": np.abs(coefficients),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    print(f"\n{'='*50}")
    print("Feature Importance (Coefficients):")
    print(f"{'='*50}")
    print(feature_importance.to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ["green" if c > 0 else "red" for c in feature_importance["Coefficient"]]
    plt.barh(
        feature_importance["Feature"], feature_importance["Coefficient"], color=colors
    )

    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Logistic Regression Feature Importance", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("logistic_regression_feature_importance.png", dpi=300)
    print(
        "\nFeature importance plot saved as 'logistic_regression_feature_importance.png'"
    )
    plt.show()


def test_regularization(X_train, X_test, y_train, y_test):
    """Test different regularization strengths."""
    print(f"\n{'='*50}")
    print("Testing Different Regularization (C values):")
    print(f"{'='*50}")

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results = []

    for C in C_values:
        lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, lr.predict(X_train))
        test_acc = accuracy_score(y_test, lr.predict(X_test))

        results.append({"C": C, "train_accuracy": train_acc, "test_accuracy": test_acc})

        print(f"\nC={C:7.3f}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    # Plot regularization effect
    plt.figure(figsize=(10, 6))
    C_vals = [r["C"] for r in results]
    train_accs = [r["train_accuracy"] for r in results]
    test_accs = [r["test_accuracy"] for r in results]

    plt.semilogx(C_vals, train_accs, marker="o", label="Training Accuracy", linewidth=2)
    plt.semilogx(C_vals, test_accs, marker="s", label="Test Accuracy", linewidth=2)

    plt.xlabel("C (Inverse Regularization Strength)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Effect of Regularization on Model Performance", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logistic_regression_regularization.png", dpi=300)
    print("\nRegularization plot saved as 'logistic_regression_regularization.png'")
    plt.show()


def cross_validation_analysis(X, y):
    """Perform cross-validation analysis."""
    print(f"\n{'='*50}")
    print("Cross-Validation Analysis (5-fold):")
    print(f"{'='*50}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")

    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


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

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling applied (StandardScaler)")

    # Train Logistic Regression
    print(f"\n{'='*50}")
    print("Training Logistic Regression:")
    print(f"{'='*50}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Evaluate on training set
    train_proba = evaluate_model(
        lr, X_train_scaled, X_test_scaled, y_train, y_test, "Training"
    )

    # Evaluate on test set
    test_proba = evaluate_model(
        lr, X_train_scaled, X_test_scaled, y_train, y_test, "Test"
    )

    # Plot ROC curve
    plot_roc_curve(y_test, test_proba)

    # Plot feature importance
    plot_feature_importance(lr, X.columns.tolist())

    # Test regularization
    test_regularization(X_train_scaled, X_test_scaled, y_train, y_test)

    # Cross-validation
    cross_validation_analysis(X, y)

    # Print model parameters
    print(f"\n{'='*50}")
    print("Model Parameters:")
    print(f"{'='*50}")
    print(f"Intercept: {lr.intercept_[0]:.4f}")
    print(f"Number of iterations: {lr.n_iter_[0]}")


if __name__ == "__main__":
    main()
