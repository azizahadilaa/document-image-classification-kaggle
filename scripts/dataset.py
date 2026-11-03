"""Dataset PyTorch dan transformasi untuk klasifikasi gambar dokumen."""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_transforms(img_size: int = 224, train: bool = True):
    """Buat pipeline transformasi gambar.

    Augmentasi ringan (tanpa flip, karena teks akan terbalik).

    Args:
        img_size: Ukuran sisi gambar input model.
        train: True untuk transformasi training, False untuk val/test.

    Returns:
        Objek transforms.Compose.
    """
    base = [T.Grayscale(num_output_channels=3), T.Resize((img_size, img_size))]
    if train:
        base += [T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
                 T.ColorJitter(brightness=0.2, contrast=0.2)]
    return T.Compose(base + [T.ToTensor(), T.Normalize(MEAN, STD)])


class DocumentDataset(Dataset):
    """Dataset gambar dokumen yang membaca JPEG hasil cache.

    Args:
        df: DataFrame dengan kolom 'id' (dan 'label' jika bukan test).
        cache_dir: Folder berisi `<id>.jpg`.
        transform: Transformasi gambar.
        is_test: Jika True, label tidak dikembalikan.
    """

    def __init__(self, df: pd.DataFrame, cache_dir: str, transform=None, is_test: bool = False):
        self.ids = df["id"].tolist()
        self.labels = None if is_test else df["label"].tolist()
        self.cache_dir, self.transform, self.is_test = cache_dir, transform, is_test

    def __len__(self) -> int:
        """Jumlah sampel."""
        return len(self.ids)

    def __getitem__(self, idx: int):
        """Ambil satu sampel (gambar, label) atau gambar saja untuk test."""
        img = Image.open(os.path.join(self.cache_dir, f"{self.ids[idx]}.jpg")).convert("L")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def make_loader(df, cache_dir, img_size, batch_size, train, is_test=False, num_workers=4):
    """Bangun DataLoader dari DataFrame.

    Args:
        df: DataFrame sumber.
        cache_dir: Folder cache gambar.
        img_size: Ukuran input model.
        batch_size: Ukuran batch.
        train: True untuk augmentasi + shuffle.
        is_test: True jika data tanpa label.
        num_workers: Jumlah worker DataLoader.

    Returns:
        torch.utils.data.DataLoader.
    """
    ds = DocumentDataset(df, cache_dir, get_transforms(img_size, train), is_test)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=True, drop_last=train)
