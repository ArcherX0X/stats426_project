"""
Baseline Models: Logistic Regression, Random Forest, SVM (LinearSVC + PCA)
Run from project root: uv run python src/baseline.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils import PLOTS_DIR, append_results, ensure_dirs, load_data, remap_labels


def plot_confusion_matrix(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {filename}")


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, class_names):
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Macro F1:      {f1:.4f}")
    print(f"  Train Time:    {train_time:.1f}s")

    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    filename = f"baseline_{slug}_cm.png"
    plot_confusion_matrix(cm, class_names, f"{name} - Confusion Matrix", filename)

    return {"accuracy": acc, "f1": f1, "time": train_time, "report": report, "plot": filename}


def main():
    ensure_dirs()
    X_train, y_train, X_test, y_test = load_data()
    y_train, y_test, class_names = remap_labels(y_train, y_test)
    num_classes = len(class_names)
    print(f"Classes: {num_classes} | Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "SVM (LinearSVC + PCA)": Pipeline(
            [
                ("pca", PCA(n_components=100, random_state=42)),
                ("svc", LinearSVC(max_iter=3000, C=1.0, random_state=42)),
            ]
        ),
    }

    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(
            name, model, X_train, y_train, X_test, y_test, class_names
        )

    # Build markdown content
    content = "### Summary\n\n"
    content += "| Model | Test Accuracy | Macro F1 | Train Time (s) |\n"
    content += "|---|---|---|---|\n"
    for name, r in results.items():
        content += f"| {name} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['time']:.1f} |\n"

    for name, r in results.items():
        content += f"\n### {name}\n\n"
        content += f"![Confusion Matrix](plots/{r['plot']})\n\n"
        content += f"```\n{r['report']}\n```\n"

    append_results("1. Baseline Models", content)
    print("\nResults written to results/results.md")


if __name__ == "__main__":
    main()
