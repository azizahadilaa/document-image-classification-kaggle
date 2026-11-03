"""Definisi model: backbone pretrained dari timm untuk fine-tuning."""
import timm
import torch.nn as nn


def build_model(name: str = "convnext_tiny", num_classes: int = 16,
                pretrained: bool = True, drop_rate: float = 0.2,
                drop_path_rate: float = 0.1) -> nn.Module:
    """Buat model klasifikasi dengan backbone pretrained.

    Args:
        name: Nama arsitektur timm (mis. 'convnext_tiny', 'efficientnet_b0', 'resnet50').
        num_classes: Jumlah kelas output.
        pretrained: Pakai bobot ImageNet atau tidak.
        drop_rate: Dropout pada classifier head.
        drop_path_rate: Stochastic depth, regularisasi untuk mengurangi overfitting.

    Returns:
        nn.Module siap dilatih.
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=drop_rate, drop_path_rate=drop_path_rate)