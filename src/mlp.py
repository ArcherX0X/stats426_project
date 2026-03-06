"""
Model 2: Multi-Layer Perceptron (MLP)
Architecture: 784 -> 512 -> 256 -> 128 -> num_classes
Run from project root: uv run python src/mlp.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.utils import PLOTS_DIR, append_results, ensure_dirs, load_data, remap_labels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def predict_all(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.append(model(X.to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def plot_curves(train_losses, val_losses, train_accs, val_accs, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle("MLP Training Curves", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {filename}")


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


def main():
    ensure_dirs()
    X_train, y_train, X_test, y_test = load_data()
    y_train, y_test, class_names = remap_labels(y_train, y_test)
    num_classes = len(class_names)
    print(f"Device: {DEVICE} | Classes: {num_classes} | Train: {len(X_train)} | Test: {len(X_test)}")

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    full_ds = TensorDataset(X_train_t, y_train_t)
    val_size = int(0.2 * len(full_ds))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

    model = MLP(784, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")

    train_time = time.time() - t0

    y_pred = predict_all(model, test_loader)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Time: {train_time:.1f}s")

    plot_curves(train_losses, val_losses, train_accs, val_accs, "mlp_curves.png")
    plot_confusion_matrix(cm, class_names, "MLP - Confusion Matrix", "mlp_cm.png")

    content = f"**Architecture:** 784 → 512 → 256 → 128 → {num_classes}\n\n"
    content += f"**Hyperparameters:** Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LR}, Optimizer=Adam, Scheduler=StepLR\n\n"
    content += "### Metrics\n\n"
    content += "| Metric | Value |\n|---|---|\n"
    content += f"| Test Accuracy | {acc:.4f} |\n"
    content += f"| Macro F1 | {f1:.4f} |\n"
    content += f"| Train Time (s) | {train_time:.1f} |\n\n"
    content += "### Training Curves\n\n"
    content += "![Training Curves](plots/mlp_curves.png)\n\n"
    content += "### Confusion Matrix\n\n"
    content += "![Confusion Matrix](plots/mlp_cm.png)\n\n"
    content += f"```\n{report}\n```\n"

    append_results("2. MLP", content)
    print("Results written to results/results.md")


if __name__ == "__main__":
    main()
