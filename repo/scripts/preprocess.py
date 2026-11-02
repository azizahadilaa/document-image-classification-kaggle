"""Pre-resize gambar .tif ke JPEG kecil agar training jauh lebih cepat."""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _convert_one(args) -> None:
    """Konversi satu gambar .tif menjadi JPEG grayscale ukuran `size`.

    Args:
        args: Tuple (src_path, dst_path, size).
    """
    src, dst, size = args
    if os.path.exists(dst):
        return
    try:
        img = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
    except Exception:
        img = Image.new("L", (size, size), 255)  # fallback: halaman putih
    img.save(dst, quality=95)


def cache_images(df: pd.DataFrame, data_dir: str, out_dir: str,
                 size: int = 224, workers: int = 8) -> None:
    """Resize semua gambar pada `df` dan simpan sebagai `<id>.jpg` di `out_dir`.

    Args:
        df: DataFrame dengan kolom 'id' dan 'filename'.
        data_dir: Folder dasar data kompetisi.
        out_dir: Folder tujuan cache.
        size: Sisi gambar hasil resize (persegi).
        workers: Jumlah thread paralel.
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(os.path.join(data_dir, f), os.path.join(out_dir, f"{i}.jpg"), size)
            for i, f in zip(df["id"], df["filename"])]
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(_convert_one, jobs), total=len(jobs)))
