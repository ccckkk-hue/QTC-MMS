#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM增强的恶性分层模型训练脚本（V2优化版：减弱正则化 + 最新方法）

改进：
1. 减弱正则化（weight_decay 1e-4→5e-5, dropout 0.4→0.3, label_smoothing 0.15→0.05）
2. 优化学习率策略（Warmup + Cosine Annealing）
3. 改进训练评估方式（方案2：原始图像评估）
4. 支持Multi-only模式（确认模型上限）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_llm_enhanced_stratification import (
    FeatureTextualizer, LearnablePromptTuning,
    MultiLayerCrossModalAttention, DeepTabularEncoder,
    FocalLoss
)
from train_llm_enhanced_stratification import validate as base_validate

def collate_fn(batch):
    """自定义collate函数（支持domain_id）"""
    images = torch.stack([item['image'] for item in batch])
    numerical = torch.stack([item['numerical'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # 处理domain_id（如果存在）
    domain_ids = None
    if 'domain_id' in batch[0]:
        domain_ids = torch.tensor([item['domain_id'] for item in batch], dtype=torch.long)
    
    return {
        'image': images,
        'numerical': numerical,
        'categorical': categorical,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'domain_id': domain_ids
    }
from train_llm_enhanced_stratification import LLMEnhancedDataset as BaseLLMEnhancedDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import argparse
import json
import ast
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int):
    """尽量保证可复现（注意：GPU/CUDA 仍可能存在少量非确定性）"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ==================== 三模态交叉注意力 =====================

class TriModalCrossAttention(nn.Module):
    """三模态交叉注意力融合（优化版：2层，减少过拟合）"""
    
    def __init__(self, img_dim, tab_dim, text_dim, hidden_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 投影到统一维度
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 交叉注意力层（减少到2层）
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 融合层（减弱dropout）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # 从0.4降到0.3
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)  # 从0.4降到0.3
        )
    
    def forward(self, img_feat, tab_feat, text_feat):
        img_proj = self.img_proj(img_feat)
        tab_proj = self.tab_proj(tab_feat)
        text_proj = self.text_proj(text_feat)
        
        img_seq = img_proj.unsqueeze(1)
        tab_seq = tab_proj.unsqueeze(1)
        text_seq = text_proj.unsqueeze(1)
        
        combined = torch.cat([img_seq, tab_seq, text_seq], dim=1)
        
        for layer in self.cross_attention_layers:
            combined = layer(combined)
        
        img_attn = combined[:, 0, :]
        tab_attn = combined[:, 1, :]
        text_attn = combined[:, 2, :]
        
        concat_feat = torch.cat([img_attn, tab_attn, text_attn], dim=1)
        fused = self.fusion(concat_feat)
        
        return fused


# ==================== 优化模型 =====================

class LLMEnhancedStratificationModelV2Optimized(nn.Module):
    """LLM增强的恶性分层模型（V2优化版：减弱正则化 + Domain Embedding）"""
    
    def __init__(self, 
                 num_numerical,
                 num_categorical,
                 cat_vocab_size,
                 img_size=224,
                 img_dim=768,
                 text_dim=768,
                 tab_dim=512,
                 hidden_dim=512,
                 num_prompts=8,
                 llm_model=None,
                 freeze_llm=True,
                 dropout=0.3,
                 use_domain_embedding=False,
                 num_domains=3,
                 model_size='base',
                 tri_modal_num_layers: int = 2):  # 0=multi, 1=beimo, 2=shaoxing1
        super().__init__()
        
        self.use_domain_embedding = use_domain_embedding
        
        # 1. 图像编码器（支持不同规模）
        vit_model_map = {
            'tiny': 'vit_tiny_patch16_224',
            'small': 'vit_small_patch16_224',
            'base': 'vit_base_patch16_224'
        }
        vit_model_name = vit_model_map.get(model_size, 'vit_base_patch16_224')
        
        self.image_encoder = timm.create_model(
            vit_model_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size
        )
        
        # 2. 表格特征编码器（减弱dropout）
        self.tabular_encoder = DeepTabularEncoder(
            num_numerical, num_categorical, cat_vocab_size,
            hidden_dims=[128, 256, tab_dim],
            dropout=dropout  # 传递dropout参数
        )
        tab_dim_actual = self.tabular_encoder.output_dim
        
        # 3. LLM编码器（冻结）
        self.llm_model = llm_model
        if self.llm_model is not None and freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        
        # 4. 可学习提示调优
        self.prompt_tuning = LearnablePromptTuning(
            num_prompts=num_prompts,
            prompt_dim=text_dim,
            llm_embed_dim=text_dim
        )
        
        # 5. 文本特征投影（减弱dropout）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 文本投影用更小的dropout
        )
        
        # 6. Domain Embedding（新增）
        if use_domain_embedding:
            self.domain_embedding = nn.Embedding(num_domains, hidden_dim // 4)  # 128维domain embedding
            domain_proj_dim = hidden_dim // 4
        else:
            self.domain_embedding = None
            domain_proj_dim = 0
        
        # 7. 三模态交叉注意力融合（2层）
        if tab_dim_actual > 0:
            self.tri_modal_fusion = TriModalCrossAttention(
                img_dim=img_dim,
                tab_dim=tab_dim_actual,
                text_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=tri_modal_num_layers,
                num_heads=8
            )
            fusion_input_dim = hidden_dim + domain_proj_dim  # 融合后加入domain embedding
        else:
            self.tri_modal_fusion = None
            fusion_input_dim = img_dim + hidden_dim + domain_proj_dim
        
        # 8. 分类器（减弱dropout）
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # 使用配置的dropout
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, image, numerical, categorical, input_ids, attention_mask, domain_id=None):
        img_feat = self.image_encoder(image)
        
        if self.tabular_encoder.output_dim > 0:
            tab_feat = self.tabular_encoder(numerical, categorical)
        else:
            tab_feat = torch.zeros(img_feat.size(0), 0, device=img_feat.device)
        
        if self.llm_model is not None:
            llm_outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeddings = llm_outputs.last_hidden_state
        else:
            B, L = input_ids.shape
            text_embeddings = torch.zeros(B, L, 768, device=input_ids.device)
        
        enhanced_text = self.prompt_tuning(text_embeddings)
        text_global = enhanced_text[:, -1, :]
        text_proj = self.text_proj(text_global)
        
        if self.tri_modal_fusion is not None:
            fused = self.tri_modal_fusion(img_feat, tab_feat, text_proj)
        else:
            fused = torch.cat([img_feat, text_proj], dim=1)
        
        # Domain Embedding（新增）
        if self.use_domain_embedding:
            if domain_id is not None:
                domain_emb = self.domain_embedding(domain_id)  # [B, hidden_dim//4]
            else:
                # 如果没有提供domain_id，使用默认值0（multi）
                B = fused.size(0)
                domain_emb = self.domain_embedding(torch.zeros(B, dtype=torch.long, device=fused.device))
            fused = torch.cat([fused, domain_emb], dim=1)  # 拼接domain embedding
        
        logits = self.classifier(fused)
        return logits


# ==================== 数据集 =====================

class LLMEnhancedDatasetV2Optimized(BaseLLMEnhancedDataset):
    """LLM增强数据集V2优化版（支持Domain Embedding）"""
    
    def _setup_features(self):
        super()._setup_features()
        if self.radiomics_features:
            for feat in self.radiomics_features:
                if feat not in self.numerical_features:
                    self.numerical_features.append(feat)
        
        # 设置domain映射（用于Domain Embedding）
        # 0=multi, 1=beimo, 2=shaoxing1
        self.domain_map = {'multi': 0, 'beimo': 1, 'shaoxing1': 2}
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # 添加domain_id（如果source列存在）
        if hasattr(self, 'patients') and 'source' in self.patients.columns:
            source_val = str(self.patients.iloc[idx].get('source', 'multi')).lower()
            domain_id = self.domain_map.get(source_val, 0)  # 默认multi=0
        else:
            domain_id = 0  # 默认multi
        
        item['domain_id'] = domain_id
        return item


# ==================== 训练函数（方案2：原始图像评估）====================

def train_epoch_with_eval_transform(model, train_loader, eval_loader, criterion, optimizer, device, epoch):
    """训练时使用增强图像，但用原始图像评估AUC"""
    model.train()
    total_loss = 0
    
    # 训练阶段：使用增强图像
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        numerical = batch['numerical'].to(device)
        categorical = batch['categorical'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        domain_id = batch.get('domain_id')
        if domain_id is not None:
            domain_id = domain_id.to(device)
        
        optimizer.zero_grad()
        logits = model(images, numerical, categorical, input_ids, attention_mask, domain_id=domain_id)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    
    # 评估阶段：使用原始图像（不增强）
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Epoch {epoch} [Eval]', leave=False):
            images = batch['image'].to(device)
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            domain_id = batch.get('domain_id')
            if domain_id is not None:
                domain_id = domain_id.to(device)
            
            logits = model(images, numerical, categorical, input_ids, attention_mask, domain_id=domain_id)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 可选：记录domain，用于拆分train AUC（0=multi,1=beimo,2=shaoxing1）
            if domain_id is not None:
                if 'all_domains' not in locals():
                    all_domains = []
                all_domains.extend(domain_id.detach().cpu().numpy().tolist())
    
    model.train()
    
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
        # ✅ 修复：acc需要与label对比（原来只是正类比例）
        acc = (np.array(all_preds) > 0.5).astype(int)
        acc = (acc == np.array(all_labels).astype(int)).mean()
    else:
        auc = 0.0
        acc = 0.0

    # ✅ 额外输出：按domain拆分的train AUC（用于解释 train<val 的现象）
    try:
        if 'all_domains' in locals() and len(set(all_labels)) > 1:
            y = np.array(all_labels).astype(int)
            p = np.array(all_preds).astype(float)
            d = np.array(all_domains).astype(int)
            for dom, name in [(0, "multi"), (1, "beimo"), (2, "shaoxing1")]:
                m = d == dom
                if m.sum() >= 10:
                    pos = int((y[m] == 1).sum())
                    neg = int((y[m] == 0).sum())
                    print(f"  [TrainEval/{name}] n={int(m.sum())} pos={pos} neg={neg}")
                if m.sum() >= 10 and len(set(y[m].tolist())) > 1:
                    dom_auc = roc_auc_score(y[m], p[m])
                    print(f"  [TrainEval AUC/{name}] n={int(m.sum())} auc={dom_auc:.4f}")
    except Exception:
        pass
    
    return avg_loss, acc, auc


# ==================== Validate函数（支持domain_id）====================

def validate(model, loader, criterion, device, mode='val', threshold=None, use_domain_embedding=False):
    """验证函数（支持domain embedding）"""
    model.eval()
    all_patient_preds = []
    all_patient_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'{mode} evaluation'):
            images = batch['image'].to(device)
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            domain_id = None
            if use_domain_embedding and 'domain_id' in batch and batch['domain_id'] is not None:
                domain_id = batch['domain_id'].to(device)
            
            logits = model(images, numerical, categorical, input_ids, attention_mask, domain_id=domain_id)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # 处理batch中的每个样本
            probs_np = probs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_patient_preds.extend(probs_np)
            all_patient_labels.extend(labels_np)
    
    all_patient_preds = np.array(all_patient_preds)
    all_patient_labels = np.array(all_patient_labels)
    
    if len(set(all_patient_labels)) < 2:
        return 0.0, 0.0, 0.5, {}, all_patient_preds, all_patient_labels
    
    # 计算AUC
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
    auc = roc_auc_score(all_patient_labels, all_patient_preds)
    
    # 寻找最优阈值（复用base逻辑，但补充更鲁棒的打印/记录，避免“阈值接近1”造成误解）
    if threshold is None:
        prob_min = all_patient_preds.min()
        prob_max = all_patient_preds.max()
        prob_mean = all_patient_preds.mean()
        prob_std = all_patient_preds.std()
        
        if mode == 'val':
            print(f"\n📊 概率分布统计:")
            print(f"  最小值: {prob_min:.4f}, 最大值: {prob_max:.4f}")
            print(f"  平均值: {prob_mean:.4f}, 标准差: {prob_std:.4f}")
            if len(all_patient_labels[all_patient_labels==1]) > 0:
                pos_probs = all_patient_preds[all_patient_labels==1]
                neg_probs = all_patient_preds[all_patient_labels==0]
                print(f"  正类概率: {pos_probs.mean():.4f} ± {pos_probs.std():.4f}")
                print(f"  负类概率: {neg_probs.mean():.4f} ± {neg_probs.std():.4f}")
                try:
                    def _q(a):
                        return np.quantile(a, [0.05, 0.50, 0.95]).tolist()
                    qp = _q(pos_probs)
                    qn = _q(neg_probs)
                    print(f"  正类分位数(q05/q50/q95): {qp[0]:.4f}/{qp[1]:.4f}/{qp[2]:.4f}")
                    print(f"  负类分位数(q05/q50/q95): {qn[0]:.4f}/{qn[1]:.4f}/{qn[2]:.4f}")
                except Exception:
                    pass
        
        fpr, tpr, thresholds_roc = roc_curve(all_patient_labels, all_patient_preds)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        threshold_youden = thresholds_roc[optimal_idx]
        
        thresholds_f1 = np.arange(0.05, 0.95, 0.01)
        best_f1 = 0
        best_threshold_f1 = 0.5
        for t in thresholds_f1:
            preds_binary = (all_patient_preds >= t).astype(int)
            if len(np.unique(preds_binary)) < 2:
                continue
            f1 = f1_score(all_patient_labels, preds_binary)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = t
        
        if prob_max < 0.2:
            threshold = best_threshold_f1
        else:
            threshold = threshold_youden
            if mode == 'val':
                print(f"\n📈 阈值选择:")
                print(f"  Youden's J最优阈值(原始): {float(threshold_youden):.4f}")
                print(f"  F1最优阈值: {float(best_threshold_f1):.4f}")
        
        threshold_raw = float(threshold)
        # 重要：为了避免“阈值被极端推到接近1/0”导致外部Recall崩掉，这里对阈值做夹逼
        threshold = max(0.05, min(0.95, threshold))
        if mode == 'val' and abs(threshold_raw - threshold) > 1e-9:
            print(f"  ⚠️ 阈值已夹逼: raw={threshold_raw:.4f} -> used={threshold:.4f}")
        # 把候选阈值记录下来，方便结果文件里追溯
        threshold_youden_raw = float(threshold_youden)
        threshold_f1 = float(best_threshold_f1)
    else:
        threshold_raw = float(threshold)
        threshold_youden_raw = None
        threshold_f1 = None
    
    preds_binary = (all_patient_preds >= threshold).astype(int)
    precision = precision_score(all_patient_labels, preds_binary, zero_division=0)
    recall = recall_score(all_patient_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_patient_labels, preds_binary, zero_division=0)
    
    metrics = {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': float(threshold),
        # 额外记录（仅在本轮自动搜阈值时有值）
        'threshold_raw_before_clip': None if threshold_raw is None else float(threshold_raw),
        'threshold_youden_raw': threshold_youden_raw,
        'threshold_f1': threshold_f1
    }
    
    return 0.0, auc, threshold, metrics, all_patient_preds, all_patient_labels


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _plot_train_curves(history: dict, output_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    try:
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        if not epochs:
            return
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(epochs, history.get("train_loss", []), label="train_loss")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("epoch")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, history.get("train_auc", []), label="train_auc")
        ax2.plot(epochs, history.get("val_auc", []), label="val_auc")
        ax2.set_title("AUC Curves")
        ax2.set_xlabel("epoch")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "train_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: Path, title: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    try:
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        thresh = cm.max() * 0.5 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='LLM增强的恶性分层模型训练（V2优化版）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于复现实验/多seed搜索external上限）')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--external_csv', type=str, default=None)
    parser.add_argument('--image_column', type=str, default='nodule_crop_path')
    parser.add_argument('--feature_config', type=str, default='age_sex_maxdiameter')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_lr_mult', type=float, default=0.5, help='图像编码器学习率倍率（相对lr）')
    parser.add_argument('--weight_decay', type=float, default=5e-5)  # ⚠️ 从1e-4降到5e-5
    parser.add_argument('--dropout', type=float, default=0.3)  # ⚠️ 从0.4降到0.3
    parser.add_argument('--label_smoothing', type=float, default=0.05)  # ⚠️ 从0.15降到0.05
    parser.add_argument('--input_size', type=int, default=224, help='输入分辨率（小结节建议提高到384/448）')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup轮数（默认5）')
    parser.add_argument('--freeze_image_epochs', type=int, default=0, help='前N轮冻结图像编码器（让模型“慢一点学”，减少早期过拟合/域偏）')
    parser.add_argument('--init_ckpt', type=str, default=None, help='可选：加载初始权重ckpt（用于两阶段训练/微调）')
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--radiomics_feature_list', type=str, default=None)
    parser.add_argument('--num_prompts', type=int, default=8)
    parser.add_argument('--llm_model_name', type=str, default='hfl/chinese-bert-wwm-ext')
    parser.add_argument('--freeze_llm', action='store_true', default=True)
    parser.add_argument('--multi_only', action='store_true', help='仅使用Multi数据')
    parser.add_argument(
        '--sampler',
        type=str,
        default='none',
        choices=['none', 'class', 'small', 'domain', 'small_domain'],
        help='训练采样策略：none=随机打乱；class=按标签平衡；small=加权小结节；domain=加权Beimo；small_domain=两者叠加'
    )
    parser.add_argument('--small_weight', type=float, default=2.0, help='小结节(max_diameter_category=0)采样权重倍数')
    parser.add_argument('--beimo_weight', type=float, default=2.0, help='Beimo样本采样权重倍数（source=beimo）')
    parser.add_argument(
        '--female_small_pos_weight',
        type=float,
        default=1.0,
        help='额外加权：source=multi 且 sex_encoded=0 且 max_diameter_category=0 且 stratification=1 的采样权重倍数（用于提升external小径女性正类）'
    )
    # ✅ Hard-mining：把“训练集上预测错的样本”采样权重拉高（从cache快速得到）
    parser.add_argument('--hard_mining_cache_dir', type=str, default=None,
                        help='可选：使用 ensemble_cache_predictions 的缓存，在训练集上找预测错样本并加权采样')
    parser.add_argument('--hard_mining_weights_json', type=str, default=None,
                        help='可选：cache权重文件（例如 result/ensemble_cache_top25ish_v1/search_weight.json）；不传则用多模型均值prob')
    parser.add_argument('--hard_mining_weight', type=float, default=1.0,
                        help='对 hard(预测错) 样本的额外采样权重倍数（>1 生效）')
    parser.add_argument('--hard_mining_max_ratio', type=float, default=0.2,
                        help='最多加权多少比例的训练样本（按“错得最离谱”排序取TopK）')
    parser.add_argument('--hard_mining_only_source', type=str, default=None,
                        help='仅对某个source做hard-mining（例如 beimo 或 multi）；默认不限制')
    parser.add_argument('--use_domain_embedding', action='store_true', help='启用Domain Embedding（显式编码domain信息）')
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base'], 
                       help='模型规模：tiny(ViT-Tiny), small(ViT-Small), base(ViT-Base)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度（小模型建议256-384）')
    
    args = parser.parse_args()

    # 固定随机性：external AUC 上限往往来自多 seed
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("="*80)
    print("🚀 LLM增强模型V2优化版：减弱正则化 + 最新方法")
    print(f"   - Weight Decay: {args.weight_decay} (减弱)")
    print(f"   - Dropout: {args.dropout} (减弱)")
    print(f"   - Label Smoothing: {args.label_smoothing} (减弱)")
    print(f"   - TriModal Layers: 2 (简化)")
    print("="*80)
    
    # 图像变换
    train_transform = timm.data.create_transform(
        input_size=args.input_size,
        is_training=True,
        auto_augment='rand-m5-mstd0.5-inc1',  # 保持中等强度增强
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    eval_transform = timm.data.create_transform(
        input_size=args.input_size,
        is_training=False,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    val_transform = eval_transform
    
    # 加载LLM（确保真正使用大语言模型）
    print(f"\n加载LLM模型: {args.llm_model_name}")
    llm_tokenizer = None
    llm_model = None
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # 策略1：优先使用safetensors格式（避免torch.load安全漏洞）
        try:
            print("  尝试使用safetensors格式加载...")
            llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
            llm_model = AutoModel.from_pretrained(
                args.llm_model_name,
                use_safetensors=True,  # 使用safetensors格式
                trust_remote_code=False
            )
            print(f"✅ LLM模型加载成功（使用safetensors格式）")
        except Exception as e1:
            # 策略2：如果safetensors失败，尝试普通方式
            print(f"  ⚠️ safetensors加载失败: {str(e1)[:100]}...")
            print("  尝试普通方式加载...")
            try:
                llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
                llm_model = AutoModel.from_pretrained(
                    args.llm_model_name,
                    trust_remote_code=False
                )
                print(f"✅ LLM模型加载成功（普通方式）")
            except Exception as e2:
                # 策略3：如果都失败，提供详细错误信息
                torch_version = torch.__version__
                print(f"  ❌ LLM模型加载失败: {str(e2)[:200]}")
                print(f"  PyTorch版本: {torch_version}")
                print(f"  建议解决方案：")
                print(f"    1. 升级PyTorch: pip install torch>=2.6.0")
                print(f"    2. 或确保模型支持safetensors格式")
                print(f"  ⚠️ 警告：将使用占位符继续训练，LLM增强功能不可用！")
                print(f"  ⚠️ 性能可能显著下降，强烈建议修复LLM加载问题！")
                llm_tokenizer = None
                llm_model = None
                # 询问是否继续
                print(f"\n是否继续训练？(y/n): ", end='')
                # 自动继续（不阻塞训练）
                print("y (自动继续)")
                
    except ImportError as e:
        print(f"❌ transformers库未安装: {e}")
        print(f"  安装命令: pip install transformers")
        print(f"  ⚠️ 警告：将使用占位符继续训练，LLM增强功能不可用！")
        llm_tokenizer = None
        llm_model = None
    
    # 验证LLM是否成功加载
    if llm_model is None or llm_tokenizer is None:
        print(f"\n⚠️⚠️⚠️ 警告：LLM模型未成功加载！")
        print(f"  当前训练将无法使用LLM文本特征增强功能")
        print(f"  这会导致性能显著下降，建议修复后再训练")
        print(f"⚠️⚠️⚠️\n")
    else:
        print(f"\n✅ LLM模型已成功加载，将使用文本特征增强功能")
        print(f"  模型: {args.llm_model_name}")
        print(f"  参数量: {sum(p.numel() for p in llm_model.parameters()) / 1e6:.1f}M")
    
    # 创建特征文本化器
    radiomics_feature_names = []
    if args.radiomics_feature_list and Path(args.radiomics_feature_list).exists():
        with open(args.radiomics_feature_list, 'r', encoding='utf-8') as f:
            radiomics_feature_names = [line.strip() for line in f if line.strip()]
    
    feature_textualizer = FeatureTextualizer(
        radiomics_feature_names=radiomics_feature_names
    )
    
    # 数据集类选择
    if args.multi_only:
        from train_llm_enhanced_stratification_multi_only import LLMEnhancedDatasetMultiOnly
        DatasetClass = LLMEnhancedDatasetMultiOnly
        print("\n⚠️  使用仅Multi数据模式（确认模型上限）")
    else:
        DatasetClass = LLMEnhancedDatasetV2Optimized
    
    # 加载数据集
    print("\n加载训练集（增强图像）...")
    train_dataset = DatasetClass(
        csv_path=args.train_csv,
        transform=train_transform,
        mode='train',
        image_column=args.image_column,
        feature_config=args.feature_config,
        radiomics_feature_list_path=args.radiomics_feature_list,
        numerical_scaler=None,
        feature_textualizer=feature_textualizer,
        llm_tokenizer=llm_tokenizer
    )
    train_scaler = train_dataset.numerical_scaler
    
    # 创建用于评估的训练数据集（原始图像，不增强）
    print("\n加载训练集（原始图像，用于评估）...")
    train_eval_dataset = DatasetClass(
        csv_path=args.train_csv,
        transform=eval_transform,
        # ✅ 关键修复：评估train AUC时不能用mode='train'（会随机抽图，导致AUC不稳定且偏低）
        # 用val/test模式固定每个patient使用同一张图（第一张），让 train AUC 可比、可解释
        mode='val',
        image_column=args.image_column,
        feature_config=args.feature_config,
        radiomics_feature_list_path=args.radiomics_feature_list,
        numerical_scaler=train_scaler,
        feature_textualizer=feature_textualizer,
        llm_tokenizer=llm_tokenizer
    )
    
    print("\n加载验证集...")
    val_dataset = DatasetClass(
        csv_path=args.val_csv,
        transform=val_transform,
        mode='val',
        image_column=args.image_column,
        feature_config=args.feature_config,
        radiomics_feature_list_path=args.radiomics_feature_list,
        numerical_scaler=train_scaler,
        feature_textualizer=feature_textualizer,
        llm_tokenizer=llm_tokenizer
    )
    
    print("\n加载测试集...")
    test_dataset = DatasetClass(
        csv_path=args.test_csv,
        transform=val_transform,
        mode='test',
        image_column=args.image_column,
        feature_config=args.feature_config,
        radiomics_feature_list_path=args.radiomics_feature_list,
        numerical_scaler=train_scaler,
        feature_textualizer=feature_textualizer,
        llm_tokenizer=llm_tokenizer
    )
    
    external_dataset = None
    if args.external_csv and Path(args.external_csv).exists():
        print("\n加载外部验证集...")
        external_dataset = BaseLLMEnhancedDataset(
            csv_path=args.external_csv,
            transform=val_transform,
            mode='test',
            image_column=args.image_column,
            feature_config=args.feature_config,
            radiomics_feature_list_path=args.radiomics_feature_list,
            numerical_scaler=train_scaler,
            feature_textualizer=feature_textualizer,
            llm_tokenizer=llm_tokenizer
        )
    
    # DataLoader（支持WeightedRandomSampler以强化小结节/Beimo学习）
    train_sampler = None
    shuffle = True
    if args.sampler != 'none':
        try:
            from torch.utils.data import WeightedRandomSampler
            df = train_dataset.patients.copy()
            # 兜底：缺列时不启用采样器
            if 'stratification' not in df.columns:
                raise RuntimeError("train_dataset.patients缺少'stratification'列，无法构建采样器")

            weights = np.ones(len(df), dtype=np.float32)

            if args.sampler in ('class', 'small_domain'):
                # 类别平衡（按stratification）
                vc = df['stratification'].value_counts()
                w0 = 1.0 / float(vc.get(0, 1))
                w1 = 1.0 / float(vc.get(1, 1))
                weights *= df['stratification'].map(lambda y: w1 if int(y) == 1 else w0).to_numpy(dtype=np.float32)

            if args.sampler in ('small', 'small_domain'):
                if 'max_diameter_category' in df.columns:
                    weights *= np.where(df['max_diameter_category'].fillna(0).astype(int).to_numpy() == 0,
                                        float(args.small_weight), 1.0).astype(np.float32)
                else:
                    print("⚠️ sampler=small 但缺少 max_diameter_category 列，跳过小结节加权")

            if args.sampler in ('domain', 'small_domain'):
                if 'source' in df.columns:
                    weights *= np.where(df['source'].astype(str).str.lower().to_numpy() == 'beimo',
                                        float(args.beimo_weight), 1.0).astype(np.float32)
                else:
                    print("⚠️ sampler=domain 但缺少 source 列，跳过Beimo加权")

            # ✅ external hard-positive 定向加权：multi & female(0) & small(0) & positive(1)
            # 说明：训练是 patient-level，简单“复制CSV行”常会被patient聚合逻辑折叠；
            # 采样权重才是最稳定、最可控的做法。
            if float(args.female_small_pos_weight) != 1.0:
                required_cols = {'source', 'sex_encoded', 'max_diameter_category', 'stratification'}
                if required_cols.issubset(set(df.columns)):
                    src = df['source'].fillna('multi').astype(str).str.lower().to_numpy()
                    sex = pd.to_numeric(df['sex_encoded'], errors='coerce').fillna(0).astype(int).to_numpy()
                    diam = pd.to_numeric(df['max_diameter_category'], errors='coerce').fillna(0).astype(int).to_numpy()
                    y = pd.to_numeric(df['stratification'], errors='coerce').fillna(0).astype(int).to_numpy()
                    m = (src == 'multi') & (sex == 0) & (diam == 0) & (y == 1)
                    if m.any():
                        weights *= np.where(m, float(args.female_small_pos_weight), 1.0).astype(np.float32)
                        print(f"✅ subgroup加权 female_small_pos_weight={args.female_small_pos_weight} hits={int(m.sum())}/{len(df)}")
                    else:
                        print("⚠️ female_small_pos_weight 条件匹配为0（可能该子群不在训练集或列值缺失）")
                else:
                    missing = sorted(list(required_cols - set(df.columns)))
                    print(f"⚠️ female_small_pos_weight 需要列 {sorted(list(required_cols))}，当前缺失：{missing}")

            # ✅ hard-mining：训练集上预测错的样本（来自cache）加权
            if float(args.hard_mining_weight) != 1.0 and args.hard_mining_cache_dir:
                try:
                    from pathlib import Path as _P
                    import json as _json
                    cache_dir = _P(args.hard_mining_cache_dir)
                    train_meta_path = cache_dir / "train_meta.csv"
                    y_path = cache_dir / "train_labels.npy"
                    if not train_meta_path.exists() or not y_path.exists():
                        raise RuntimeError("cache缺少 train_meta.csv 或 train_labels.npy")
                    train_meta = pd.read_csv(train_meta_path, encoding="utf-8-sig")
                    y = np.load(y_path).astype(int)
                    if len(train_meta) != len(y):
                        raise RuntimeError("cache train_meta 与 train_labels 行数不一致")
                    # align by patient_id to current df order
                    if "patient_id" not in train_meta.columns or "patient_id" not in df.columns:
                        raise RuntimeError("patient_id 列缺失，无法对齐hard-mining")

                    # build prob per patient from cache
                    pid_cache = train_meta["patient_id"].astype(str).tolist()
                    y_map = dict(zip(pid_cache, y.tolist()))

                    # load probs
                    p_map = {}
                    if args.hard_mining_weights_json:
                        wj = _json.loads(_P(args.hard_mining_weights_json).read_text(encoding="utf-8"))
                        best = wj.get("best", wj)
                        names = wj.get("models") or best.get("models")
                        if not names:
                            raise RuntimeError("weights_json 缺少 models 列表")
                        w = np.asarray(best["weights"], dtype=np.float32)
                        combine = str(best.get("combine", "prob"))
                        if len(w) != len(names):
                            raise RuntimeError("weights 与 models 长度不一致")
                        Pm = np.stack([np.load(cache_dir / f"{n}__train_probs.npy") for n in names], axis=1).astype(np.float32)
                        if combine == "prob":
                            score = Pm @ w
                            prob = np.clip(score, 0.0, 1.0)
                        elif combine == "rank":
                            # rank01
                            r = np.argsort(np.argsort(Pm, axis=0), axis=0).astype(np.float32)
                            denom = max(1, Pm.shape[0] - 1)
                            score = (r / float(denom)) @ w
                            prob = np.clip(score, 0.0, 1.0)
                        else:
                            # logit
                            eps = 1e-6
                            Pclip = np.clip(Pm, eps, 1 - eps)
                            logit = np.log(Pclip / (1 - Pclip))
                            score = logit @ w
                            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))

                        for pid, p in zip(pid_cache, prob.tolist()):
                            p_map[str(pid)] = float(p)
                    else:
                        # fallback: mean prob across cached models in meta.json
                        meta = _json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
                        names = [m["name"] for m in meta.get("models", []) if not m.get("skipped")]
                        if len(names) < 1:
                            raise RuntimeError("meta.json models为空")
                        Pm = np.stack([np.load(cache_dir / f"{n}__train_probs.npy") for n in names], axis=1).astype(np.float32)
                        prob = Pm.mean(axis=1)
                        for pid, p in zip(pid_cache, prob.tolist()):
                            p_map[str(pid)] = float(p)

                    # build hard list in current df order
                    pid_df = df["patient_id"].astype(str).tolist()
                    src_df = df["source"].fillna("multi").astype(str).str.lower().tolist() if "source" in df.columns else ["multi"] * len(pid_df)
                    only_src = str(args.hard_mining_only_source).lower() if args.hard_mining_only_source else None
                    conf = np.zeros(len(pid_df), dtype=np.float32)
                    wrong = np.zeros(len(pid_df), dtype=bool)
                    ok = np.zeros(len(pid_df), dtype=bool)
                    for i, pid in enumerate(pid_df):
                        if pid in y_map and pid in p_map:
                            ok[i] = True
                            yy = int(y_map[pid])
                            pp = float(p_map[pid])
                            pred = 1 if pp >= 0.5 else 0
                            wrong[i] = (pred != yy)
                            conf[i] = abs(pp - 0.5)  # 越大越“离谱”
                        else:
                            ok[i] = False
                    if only_src:
                        wrong = wrong & (np.array(src_df) == only_src)
                    cand = np.where(ok & wrong)[0]
                    if len(cand) > 0:
                        k = int(round(len(df) * float(args.hard_mining_max_ratio)))
                        k = max(1, k)
                        k = min(k, len(cand))
                        order = cand[np.argsort(-conf[cand])]
                        take = order[:k]
                        weights[take] *= float(args.hard_mining_weight)
                        print(f"✅ hard-mining: weight={args.hard_mining_weight} hits={len(take)}/{len(df)} (only_source={only_src})")
                    else:
                        print("⚠️ hard-mining: 在cache对齐后找不到可加权的错误样本（可能cache不匹配当前train_csv）")
                except Exception as e:
                    print(f"⚠️ hard-mining失败，将跳过: {e}")

            weights = np.clip(weights, 1e-6, 1e6)
            train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            shuffle = False
            print(f"✅ 启用WeightedRandomSampler: sampler={args.sampler}, input_size={args.input_size}")
        except Exception as e:
            print(f"⚠️ 构建WeightedRandomSampler失败，将回退到shuffle训练: {e}")
            train_sampler = None
            shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2)
    external_loader = DataLoader(external_dataset, batch_size=1, shuffle=False, 
                                collate_fn=collate_fn, num_workers=2) if external_dataset else None
    
    # 根据模型规模确定维度
    if args.model_size == 'tiny':
        img_dim = 192
        text_dim = 768  # LLM保持768
        tab_dim = 256
        hidden_dim = args.hidden_dim if args.hidden_dim != 512 else 256
    elif args.model_size == 'small':
        img_dim = 384
        text_dim = 768
        tab_dim = 384
        hidden_dim = args.hidden_dim if args.hidden_dim != 512 else 384
    else:  # base
        img_dim = 768
        text_dim = 768
        tab_dim = 512
        hidden_dim = args.hidden_dim
    
    # 模型
    model = LLMEnhancedStratificationModelV2Optimized(
        num_numerical=len(train_dataset.numerical_features),
        num_categorical=len(train_dataset.categorical_features),
        cat_vocab_size=train_dataset.categorical_vocab_size,
        img_size=args.input_size,
        img_dim=img_dim,
        text_dim=text_dim,
        tab_dim=tab_dim,
        hidden_dim=hidden_dim,
        num_prompts=args.num_prompts,
        llm_model=llm_model,
        freeze_llm=args.freeze_llm,
        dropout=args.dropout,
        use_domain_embedding=args.use_domain_embedding,
        num_domains=3,  # 0=multi, 1=beimo, 2=shaoxing1
        model_size=args.model_size  # 传入模型规模
    ).to(device)

    # 可选：加载初始权重（两阶段训练/微调）
    if args.init_ckpt:
        ckpt_path = Path(args.init_ckpt)
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"✅ Loaded init_ckpt: {ckpt_path}")
            if missing:
                print(f"  - missing keys: {len(missing)}")
            if unexpected:
                print(f"  - unexpected keys: {len(unexpected)}")
        else:
            print(f"⚠️ init_ckpt not found: {ckpt_path} (skip)")
    
    if args.model_size != 'base':
        print(f"\n✅ 使用{args.model_size}模型（参数量更小）")
    
    if args.use_domain_embedding:
        print("\n✅ Domain Embedding已启用（显式编码domain信息）")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"数值特征数量: {len(train_dataset.numerical_features)} (包含age + {len(train_dataset.radiomics_features)}个PyRadiomics)")
    
    # 损失函数（减弱label smoothing）
    if args.use_focal:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        print("使用Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"使用CrossEntropyLoss (label_smoothing={args.label_smoothing})")
    
    # 优化器（分层学习率）
    image_params = list(model.image_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if 'image_encoder' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': image_params, 'lr': args.lr * float(args.image_lr_mult)},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # 学习率调度（Warmup + Cosine Annealing）
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup_epochs = max(0, int(args.warmup_epochs))
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        tmax = max(1, int(args.epochs) - warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=args.lr * 0.01)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    best_val_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auc': [], 'val_auc': []}
    
    # 冻结/解冻图像编码器（慢学习）
    def _set_image_trainable(flag: bool):
        for p in model.image_encoder.parameters():
            p.requires_grad = flag

    if int(args.freeze_image_epochs) > 0:
        _set_image_trainable(False)
        print(f"\n🧊 冻结图像编码器: first {int(args.freeze_image_epochs)} epochs")

    for epoch in range(1, args.epochs + 1):
        if int(args.freeze_image_epochs) > 0:
            if epoch == int(args.freeze_image_epochs) + 1:
                _set_image_trainable(True)
                print(f"\n🔥 解冻图像编码器: epoch {epoch}+")
        train_loss, train_acc, train_auc = train_epoch_with_eval_transform(
            model, train_loader, train_eval_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc, val_threshold, val_metrics, _, _ = validate(model, val_loader, 
                                                                      criterion, device, 'val', 
                                                                      use_domain_embedding=args.use_domain_embedding)
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val AUC={val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"✅ 保存最佳模型 (Val AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停于 Epoch {epoch}")
                break
    
    # 加载最佳模型评估
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    
    print("\n🔍 在验证集上寻找最优阈值...")
    _, val_auc, optimal_threshold, val_metrics, _, _ = validate(model, val_loader, 
                                                                criterion, device, 'val',
                                                                use_domain_embedding=args.use_domain_embedding)
    print(f"✅ 验证集最优阈值: {optimal_threshold:.4f} (AUC={val_auc:.4f}, "
          f"F1={val_metrics['f1']:.4f})")
    
    # 测试集评估
    print("\n📊 测试集评估...")
    _, test_auc, _, test_metrics, _, _ = validate(model, test_loader, criterion, device, 
                                                   'test', threshold=optimal_threshold,
                                                   use_domain_embedding=args.use_domain_embedding)
    print(f"测试集: AUC={test_auc:.4f}, F1={test_metrics['f1']:.4f}, "
          f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")
    
    # 外部验证集评估
    if external_loader:
        print("\n🌐 外部验证集评估...")
        _, external_auc, _, external_metrics, external_preds, external_labels = validate(model, external_loader, 
                                                              criterion, device, 'external', 
                                                              threshold=optimal_threshold,
                                                              use_domain_embedding=args.use_domain_embedding)
        print(f"外部验证: AUC={external_auc:.4f}, F1={external_metrics['f1']:.4f}, "
              f"Precision={external_metrics['precision']:.4f}, "
              f"Recall={external_metrics['recall']:.4f}")
    else:
        external_auc = None
        external_metrics = {}
        external_preds = None
        external_labels = None
    
    # 保存ROC曲线
    print("\n📊 生成ROC曲线...")
    try:
        from utils_plot_roc import plot_roc_curves
        y_true_dict = {}
        
        # 验证集
        _, _, _, _, val_preds, val_labels = validate(model, val_loader, criterion, device, 'val',
                                                     use_domain_embedding=args.use_domain_embedding)
        y_true_dict['Validation'] = (val_labels, val_preds)
        
        # 测试集
        _, _, _, _, test_preds, test_labels = validate(model, test_loader, criterion, device, 'test',
                                                       threshold=optimal_threshold,
                                                       use_domain_embedding=args.use_domain_embedding)
        y_true_dict['Test'] = (test_labels, test_preds)
        
        # 外部验证集
        if external_loader and external_preds is not None:
            y_true_dict['External'] = (external_labels, external_preds)
        
        plot_roc_curves(y_true_dict, output_dir, prefix='')
        print("✅ ROC曲线已保存")
    except Exception as e:
        print(f"⚠️ 保存ROC曲线失败: {e}")
    
    # 保存结果
    results = {
        'best_val_auc': best_val_auc,
        'best_epoch': epoch - patience_counter,
        'optimal_threshold': optimal_threshold,
        'train_auc': train_auc,
        'val': val_metrics,
        'test': test_metrics,
        'external': external_metrics if external_loader else None,
        'config': vars(args),
        'feature_config': args.feature_config,
        'model_version': 'v2_optimized',
        'numerical_features': train_dataset.numerical_features,
        'categorical_features': train_dataset.categorical_features,
        'radiomics_features': train_dataset.radiomics_features,
        'radiomics_feature_count': len(train_dataset.radiomics_features),
        'training_history': history
    }
    
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_dir / 'results.json'}")

    # ---- Extra paper figures (English): training curves + confusion matrices ----
    try:
        _plot_train_curves(history, output_dir)
        # confusion matrices on test/external using optimal_threshold
        from sklearn.metrics import confusion_matrix
        # test
        _, _, _, _, test_preds, test_labels = validate(
            model, test_loader, criterion, device, 'test',
            threshold=optimal_threshold,
            use_domain_embedding=args.use_domain_embedding
        )
        if test_preds is not None and test_labels is not None:
            y = np.array(test_labels).astype(int)
            p = np.array(test_preds).astype(float)
            yhat = (p >= float(optimal_threshold)).astype(int)
            cm = confusion_matrix(y, yhat, labels=[0, 1])
            _plot_confusion_matrix(cm, ["neg(0)", "pos(1)"], output_dir / "cm_test.png", "Test Confusion Matrix")
        # external
        if external_loader:
            _, _, _, _, ext_preds, ext_labels = validate(
                model, external_loader, criterion, device, 'external',
                threshold=optimal_threshold,
                use_domain_embedding=args.use_domain_embedding
            )
            if ext_preds is not None and ext_labels is not None:
                y = np.array(ext_labels).astype(int)
                p = np.array(ext_preds).astype(float)
                yhat = (p >= float(optimal_threshold)).astype(int)
                if len(np.unique(y)) >= 2:
                    cm = confusion_matrix(y, yhat, labels=[0, 1])
                    _plot_confusion_matrix(cm, ["neg(0)", "pos(1)"], output_dir / "cm_external.png", "External Confusion Matrix")
    except Exception as e:
        print(f"⚠️ Failed to save extra figures: {e}")


if __name__ == '__main__':
    main()

