"""Fungsi bantu umum: seed, deteksi path data, dan plotting."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed agar hasil eksperimen reproducible.

    Args:
        seed: Nilai seed untuk random, numpy, dan torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_dir(root: str = "/kaggle/input") -> str:
    """Cari folder yang berisi train.csv di bawah `root`.

    Args:
        root: Folder awal pencarian (default input Kaggle).

    Returns:
        Path folder yang memuat train.csv.

    Raises:
        FileNotFoundError: Jika train.csv tidak ditemukan.
    """
    for dirpath, _, files in os.walk(root):
        if "train.csv" in files:
            return dirpath
    raise FileNotFoundError(f"train.csv tidak ditemukan di {root}")


def plot_history(history: dict, save_path: str = None) -> None:
    """Plot kurva loss dan akurasi training vs validasi.

    Args:
        history: Dict berisi list 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Jika diisi, gambar disimpan ke path ini.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = None) -> None:
    """Plot confusion matrix ternormalisasi per baris.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Daftar nama kelas berurutan sesuai label.
        save_path: Jika diisi, gambar disimpan ke path ini.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediksi"); plt.ylabel("Sebenarnya")
    plt.title("Confusion Matrix (normalized)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
