"""Pre-resize gambar .tif ke JPEG kecil agar training jauh lebih cepat."""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
from tqdm import tqdm

from scripts.utils import resolve_path


def _convert_one(args) -> bool:
    """Konversi satu gambar .tif menjadi JPEG grayscale ukuran `size`.

    Args:
        args: Tuple (src_path, dst_path, size).

    Returns:
        True jika berhasil dibaca, False jika gagal (disimpan halaman putih).
    """
    src, dst, size = args
    if os.path.exists(dst):
        return True
    ok = True
    try:
        img = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
    except Exception:
        img = Image.new("L", (size, size), 255)  # fallback: halaman putih
        ok = False
    img.save(dst, quality=95)
    return ok


def cache_images(df: pd.DataFrame, data_dir: str, out_dir: str,
                 size: int = 224, workers: int = 8, max_fail_ratio: float = 0.01) -> None:
    """Resize semua gambar pada `df` dan simpan sebagai `<id>.jpg` di `out_dir`.

    Args:
        df: DataFrame dengan kolom 'id' dan 'filename'.
        data_dir: Folder dasar data kompetisi.
        out_dir: Folder tujuan cache.
        size: Sisi gambar hasil resize (persegi).
        workers: Jumlah thread paralel.
        max_fail_ratio: Batas proporsi gambar gagal sebelum proses dihentikan dengan error.

    Raises:
        RuntimeError: Jika proporsi gambar gagal dibaca melebihi `max_fail_ratio`.
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(resolve_path(data_dir, f), os.path.join(out_dir, f"{i}.jpg"), size)
            for i, f in zip(df["id"], df["filename"])]
    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(_convert_one, jobs), total=len(jobs)))
    n_fail = results.count(False)
    print(f"Selesai: {len(jobs) - n_fail} berhasil, {n_fail} gagal")
    if n_fail / len(jobs) > max_fail_ratio:
        raise RuntimeError(f"{n_fail} gambar gagal dibaca. Cek path data. Hapus folder cache sebelum mengulang.")