import os
import yaml
import torch
import pytorch_lightning as pl
import numpy as np

from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



class ArtifactMapTransform:
    def __init__(self, blur_radius=2):
        self.blur_radius = blur_radius

    def __call__(self, img):
        img = img.convert("RGB")

        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        img_np = np.array(img).astype(np.float32)
        blurred_np = np.array(blurred).astype(np.float32)

        artifact = np.abs(img_np - blurred_np)
        artifact = artifact / (artifact.max() + 1e-8) * 255
        artifact = artifact.astype(np.uint8)

        return Image.fromarray(artifact)


def build_transforms(input_mode="rgb"):
    if input_mode not in ["rgb", "artifact"]:
        raise ValueError("input_mode must be 'rgb' or 'artifact'")

    if input_mode == "rgb":
        train_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(380),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        train_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(380),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ArtifactMapTransform(blur_radius=2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform = transforms.Compose([
            transforms.Resize((380, 380)),
            ArtifactMapTransform(blur_radius=2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return train_transform, val_transform



with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

input_mode = cfg.get("input_mode", "rgb")
checkpoint_filename = cfg.get("checkpoint_filename", "best_model")

print("=" * 60)
print("Backbone model: EfficientNet-B4")
print(f"Input mode: {input_mode}")
print(f"Checkpoint filename: {checkpoint_filename}.ckpt")
print("Input resolution: 380x380")
print("Patch mode: False")
print("=" * 60)

train_transform, val_transform = build_transforms(input_mode)


train_sources = [(p, None) for p in cfg["train_paths"]]
val_sources = [(p, None) for p in cfg["val_paths"]]



train_dataset = HybridDeepfakeDataset(train_sources, transform=train_transform)
val_dataset = HybridDeepfakeDataset(val_sources, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=0
)



weights = EfficientNet_B4_Weights.IMAGENET1K_V1
backbone = efficientnet_b4(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features, 2)
)

model = DeepfakeDetector(backbone, lr=cfg["lr"])



monitor_metric = cfg.get("monitor_metric", "val_loss")
monitor_mode = cfg.get("monitor_mode", "min")

checkpoint = ModelCheckpoint(
    monitor=monitor_metric,
    dirpath="models",
    filename=checkpoint_filename,
    save_top_k=1,
    mode=monitor_mode,
    save_weights_only=True
)

early_stop = EarlyStopping(
    monitor=monitor_metric,
    patience=cfg.get("early_stopping_patience", 4),
    mode=monitor_mode
)


trainer = pl.Trainer(
    max_epochs=cfg["num_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint, early_stop],
    enable_progress_bar=True,
    log_every_n_steps=cfg.get("log_every_n_steps", 1),
    logger=cfg.get("logger", False)
)


trainer.fit(model, train_loader, val_loader)
