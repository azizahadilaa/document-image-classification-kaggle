"""Loop training dan validasi (PyTorch, mixed precision, checkpoint)."""
import math
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def get_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Scheduler linear warmup lalu cosine decay.

    Args:
        optimizer: Optimizer yang dijadwalkan.
        total_steps: Total langkah training.
        warmup_steps: Langkah warmup di awal.

    Returns:
        LambdaLR scheduler (dipanggil per step).
    """
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """Latih model satu epoch.

    Returns:
        Tuple (rata-rata loss, akurasi).
    """
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluasi model pada data validasi.

    Returns:
        Tuple (loss, akurasi, prediksi, label sebenarnya).
    """
    model.eval()
    loss_sum, n, preds, gts = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    acc = sum(p == g for p, g in zip(preds, gts)) / n
    return loss_sum / n, acc, preds, gts


def fit(model, train_loader, val_loader, device, ckpt_dir="checkpoint",
        epochs=10, lr=2e-4, weight_decay=0.05, label_smoothing=0.1,
        warmup_epochs=1, tag="model"):
    """Training penuh dengan AdamW, cosine LR, AMP, dan penyimpanan checkpoint terbaik.

    Args:
        model: Model PyTorch.
        train_loader: DataLoader training.
        val_loader: DataLoader validasi.
        device: 'cuda' atau 'cpu'.
        ckpt_dir: Folder penyimpanan checkpoint.
        epochs: Jumlah epoch.
        lr: Learning rate puncak.
        weight_decay: Weight decay AdamW.
        label_smoothing: Label smoothing pada CrossEntropy.
        warmup_epochs: Jumlah epoch warmup.
        tag: Nama file checkpoint (`<tag>_best.pt`).

    Returns:
        Dict history (train/val loss & acc per epoch).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, total, warmup_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda")
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    for ep in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"Epoch {ep+1}/{epochs} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}_best.pt"))
            print(f"  -> checkpoint terbaik disimpan (val acc {best:.4f})")
    return hist
