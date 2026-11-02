"""Inference pada data test (dengan TTA opsional) dan pembuatan file submission."""
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def predict_proba(model, loader, device, tta: bool = True) -> torch.Tensor:
    """Hitung probabilitas softmax untuk seluruh data pada `loader`.

    TTA: rata-rata prediksi gambar asli dan versi zoom-in 10%.

    Args:
        model: Model terlatih.
        loader: DataLoader tanpa label (is_test=True) tanpa shuffle.
        device: 'cuda' atau 'cpu'.
        tta: Aktifkan test-time augmentation.

    Returns:
        Tensor probabilitas ukuran (N, num_classes).
    """
    model.eval().to(device)
    outs = []
    for x in tqdm(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p = F.softmax(model(x).float(), dim=1)
            if tta:
                h, w = x.shape[-2:]
                m = int(h * 0.05)
                z = F.interpolate(x[:, :, m:h - m, m:w - m], size=(h, w),
                                  mode="bilinear", align_corners=False)
                p = (p + F.softmax(model(z).float(), dim=1)) / 2
        outs.append(p.cpu())
    return torch.cat(outs)


def make_submission(ids, probs: torch.Tensor, sample_path: str, out_path: str) -> pd.DataFrame:
    """Buat submission.csv mengikuti format sample_submission.

    Args:
        ids: Daftar id test (urutan sama dengan probs).
        probs: Probabilitas hasil prediksi.
        sample_path: Path sample_submission.csv.
        out_path: Path output submission.csv.

    Returns:
        DataFrame submission.
    """
    sub = pd.read_csv(sample_path)
    pred = dict(zip(ids, probs.argmax(1).tolist()))
    sub["label"] = sub["id"].map(pred).astype(int)
    sub.to_csv(out_path, index=False)
    return sub
