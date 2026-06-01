import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights, vit_b_16, ViT_B_16_Weights

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# === Load YAML config ===
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# === 🛠️ 변경 1: 학습용 스파르타 필터 (응용력 키우기) ===
train_transform = transforms.Compose([
    transforms.Resize((256, 256)), # 약간 크게 늘렸다가
    transforms.RandomCrop(224),    # 224 사이즈로 무작위 자르기 (다양한 구도)
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 거울처럼 좌우 반전
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # 밝기와 대비 살짝 흔들기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 🛠️ 변경 2: 검증용 정직한 필터 (시험은 원본 그대로!) ===
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataset Paths (하위 폴더 구조에 맞게 경로 세분화) ===
train_sources = [
    ("C:/Users/Konyang/최종 베이스라인 코드/data/train/real", 0),
    ("C:/Users/Konyang/최종 베이스라인 코드/data/train/fake", 1)
]

val_sources = [
    ("C:/Users/Konyang/최종 베이스라인 코드/data/validation/real", 0),
    ("C:/Users/Konyang/최종 베이스라인 코드/data/validation/fake", 1)
]

# === Datasets & Loaders ===
train_dataset = HybridDeepfakeDataset(train_sources, transform=train_transform)
val_dataset = HybridDeepfakeDataset(val_sources, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

# === Model Architecture (3번 모델: Vision Transformer 설정) ===
weights = ViT_B_16_Weights.IMAGENET1K_V1
backbone = vit_b_16(weights=weights)
features = backbone.heads.head.in_features
backbone.heads.head = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(features, 2)
)

model = DeepfakeDetector(backbone, lr=cfg["learning_rate"])

# === Callbacks ===
checkpoint = ModelCheckpoint(
    monitor=cfg.get("monitor_metric", "val_loss"),
    dirpath="models",
    filename="vit_best", # 체크포인트 이름 변경
    save_top_k=1,
    mode="min"
)

# 🛠️ 변경 4: 조기 종료 기준을 config.yaml에서 똑똑하게 읽어오도록 수정
early_stop = EarlyStopping(
    monitor=cfg.get("monitor_metric", "val_loss"),
    patience=cfg.get("early_stopping_patience", 4),
    mode="min"
)

# === Trainer ===
trainer = pl.Trainer(
    max_epochs=cfg["num_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint, early_stop],
    enable_progress_bar=True,
    log_every_n_steps=cfg.get("log_every_n_steps", 1)
)

# === Start Training ===
trainer.fit(model, train_loader, val_loader)