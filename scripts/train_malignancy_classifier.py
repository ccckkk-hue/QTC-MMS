#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
甲状腺良恶性分类训练脚本（图像CNN + 掩膜形态特征）
输入：data/Mask/clean/classification_list.csv （包含 image_path, view, malignancy, has_mask）
图像根目录：相对路径已在CSV中，掩膜路径通过将 img 路径中的 "fov/img" 替换为 "fov/msk" 获得
输出：模型、训练曲线、混淆矩阵、指标JSON，保存到 result/malignancy_cls/
"""

import os
import json
import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 抑制OpenCV冗余日志（不同版本API兼容）
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        try:
            cv2.setLogLevel(0)
        except Exception:
            pass


def compute_mask_features(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros(8, dtype=np.float32)

    # 二值化
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    _, bw = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    h, w = bw.shape
    area_img = float(h * w)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(8, dtype=np.float32)
    cnt = max(cnts, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, bw_w, bw_h = cv2.boundingRect(cnt)
    rect_area = float(bw_w * bw_h)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else 0.0

    # 形态学指标
    area_ratio = area / max(area_img, 1.0)
    rect_fill = area / max(rect_area, 1.0)
    circularity = 4.0 * math.pi * area / max(perim * perim, 1e-6)
    solidity = area / max(hull_area, 1e-6) if hull_area > 0 else 0.0
    aspect = bw_w / max(bw_h, 1.0)

    # 椭圆拟合离心率
    ecc = 0.0
    if len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        a = max(MA, ma) / 2.0
        b = min(MA, ma) / 2.0
        if a > 1e-6:
            ecc = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))

    # Hu 矩的前两个对数
    hu = cv2.HuMoments(cv2.moments(bw)).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    hu1 = float(hu[0])
    hu2 = float(hu[1])

    return np.array([
        area_ratio, rect_fill, circularity, solidity, aspect, ecc, hu1, hu2
    ], dtype=np.float32)


class MalignancyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str = 'image_path', label_col: str = 'malignancy', transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_mask_path(self, img_path: str) -> str:
        # data\Mask\fov\img\xxx.jpg -> data\Mask\fov\msk\xxx.jpg
        return img_path.replace("fov\\img", "fov\\msk").replace("fov/img", "fov/msk")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        label = int(row[self.label_col])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = self._resolve_mask_path(img_path)
        mask = None
        if Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        feats = compute_mask_features(mask)

        if self.transform is not None:
            img = self.transform(img)

        feats = torch.from_numpy(feats).float()
        return img, feats, label


class CNNWithMorphology(nn.Module):
    def __init__(self, backbone: str = 'efficientnet_b0', num_features: int = 8, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            in_feats = model.classifier[1].in_features
            self.cnn = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cnn_head = nn.Linear(in_feats, 256)
        else:
            model = models.resnet18(pretrained=pretrained)
            in_feats = model.fc.in_features
            self.cnn = nn.Sequential(*(list(model.children())[:-2]))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cnn_head = nn.Linear(in_feats, 256)

        self.feat_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x_img)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.cnn_head(x)
        f = self.feat_head(x_feat)
        out = self.classifier(torch.cat([x, f], dim=1))
        return out


def plot_curves(history: Dict[str, Any], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss'); plt.tight_layout()
    plt.savefig(str(save_dir / 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('acc'); plt.tight_layout()
    plt.savefig(str(save_dir / 'acc_curve.png'))
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for imgs, feats, labels in loader:
        imgs = imgs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs, feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / max(total, 1), running_corrects / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_labels = []
    all_probs = []

    for imgs, feats, labels in loader:
        imgs = imgs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        logits = model(imgs, feats)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += imgs.size(0)
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    auc = float(roc_auc_score(all_labels, all_probs)) if all_labels.size > 0 else 0.0
    return running_loss / max(total, 1), running_corrects / max(total, 1), auc


def main():
    import argparse
    parser = argparse.ArgumentParser(description='甲状腺良恶性分类训练（图像+形态）')
    parser.add_argument('--csv', type=str, default='data/Mask/clean/classification_list.csv')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--outdir', type=str, default='result/malignancy_cls')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    # 筛选有效标签（0/1），并去除缺失图像
    df = df[df['malignancy'].isin([0, 1])].copy()
    df = df[df['image_path'].apply(lambda p: Path(p).exists())]

    # 分层划分
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['malignancy']
    )
    print(f"训练: {len(train_df)}, 验证: {len(val_df)}")

    # 变换
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(args.imgsz, args.imgsz), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value='random')
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    train_ds = MalignancyDataset(train_df, transform=data_transforms['train'])
    val_ds = MalignancyDataset(val_df, transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = CNNWithMorphology(backbone=args.backbone, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_auc'].append(va_auc)

        print(f"  train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} auc={va_auc:.4f}")

        # 早停逻辑：监控验证loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # 保存最佳（基于验证准确率）
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # 保存完整模型（包含架构）
            torch.save(model, str(outdir / 'best_complete.pth'))
            # 同时保存权重（兼容性）
            torch.save(model.state_dict(), str(outdir / 'best.pt'))

        # 早停检查
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # 期末评估与可视化
    plot_curves(history, outdir)

    # 计算混淆矩阵与报告
    model.load_state_dict(torch.load(str(outdir / 'best.pt'), map_location=device))
    model.eval()
    all_labels, all_preds = [], []
    for imgs, feats, labels in val_loader:
        imgs = imgs.to(device)
        feats = feats.to(device)
        logits = model(imgs, feats)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, digits=4)

    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(str(outdir / 'confusion_matrix.png'))
    plt.close()

    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_val_acc': float(best_val_acc),
            'history': history,
            'classification_report': report
        }, f, ensure_ascii=False, indent=2)

    print('✅ 分类训练完成，结果保存在:', str(outdir))


if __name__ == '__main__':
    main()


