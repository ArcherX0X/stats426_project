import os
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.md")


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_test.csv"))

    X_train = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
    y_train = train_df["label"].values.astype(np.int64)

    X_test = test_df.drop("label", axis=1).values.astype(np.float32) / 255.0
    y_test = test_df["label"].values.astype(np.int64)

    return X_train, y_train, X_test, y_test


def remap_labels(y_train, y_test):
    """Remap sparse label indices (e.g. 0-25 minus J,Z) to consecutive 0-N."""
    classes = sorted(np.unique(y_train))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_train_r = np.array([label_to_idx[l] for l in y_train])
    y_test_r = np.array([label_to_idx[l] for l in y_test])
    class_names = [chr(65 + c) for c in classes]
    return y_train_r, y_test_r, class_names


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("# ASL Sign Classification - Model Results\n\n")


def append_results(section: str, content: str):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"## {section}\n")
        f.write(f"*Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(content)
        f.write("\n---\n\n")
