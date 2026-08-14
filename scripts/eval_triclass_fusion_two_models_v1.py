#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse 2 separately-trained models into a hierarchical tri-class system:
  - Malignancy model: P(malignant)
  - High-risk stratification model (malignant-only): P(high | malignant)

Tri-class probabilities:
  P0(benign)      = 1 - Pm
  P1(mal_low)     = Pm * (1 - Ph)
  P2(mal_high)    = Pm * Ph

Outputs:
  - results.json with val/test/external metrics
  - ROC curves (val/test malignancy, val/test/external highrisk)
  - Confusion matrices (val/test tri-class)
  - Optionally calibrated fusion (Platt scaling on val) for Pm and Ph
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def _to_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        out: List[str] = []
        for e in x:
            out.extend(_to_list(e))
        return out
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                return _to_list(v)
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def _norm_path(s: str) -> str:
    return str(s).strip().strip("'\"").replace("\\", "/")


def _safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception:
        return False


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_png: Path, title: str) -> None:
    if not _try_import_matplotlib():
        return
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true.astype(int), y_score.astype(float))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    except Exception:
        pass


def _plot_cm(cm: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    if not _try_import_matplotlib():
        return
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.6, 5.0))
        plt.imshow(cm, cmap="Blues")
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, int(v), ha="center", va="center")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    except Exception:
        pass


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def apply(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * _logit(p) + self.b)


def _fit_platt(y: np.ndarray, p: np.ndarray) -> Optional[PlattCalibrator]:
    try:
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y).astype(int)
        x = _logit(np.asarray(p).astype(float)).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            return None
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(x, y)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        return PlattCalibrator(a=a, b=b)
    except Exception:
        return None


def _load_malignancy_model(mal_ckpt: Path, backbone: str, device: str):
    import torch

    sd_raw = torch.load(str(mal_ckpt), map_location=device)
    # support checkpoint wrappers
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
        sd = sd_raw["state_dict"]
    elif isinstance(sd_raw, dict) and "model_state_dict" in sd_raw and isinstance(sd_raw["model_state_dict"], dict):
        sd = sd_raw["model_state_dict"]
    else:
        sd = sd_raw

    keys = list(sd.keys()) if isinstance(sd, dict) else []
    use_v3 = any(k.startswith("backbone.") for k in keys) or any(k.startswith("morphology_fc.") for k in keys)

    # Import the right script to guarantee same architecture.
    script = Path(__file__).parent / ("train_malignancy_classifier_v3.py" if use_v3 else "train_malignancy_classifier.py")
    spec = importlib.util.spec_from_file_location("train_malignancy_classifier_auto", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load: {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    CNNWithMorphology = getattr(mod, "CNNWithMorphology")
    compute_mask_features = getattr(mod, "compute_mask_features")
    model = CNNWithMorphology(backbone=backbone, pretrained=False).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, compute_mask_features


@torch.no_grad()  # type: ignore[name-defined]
def _predict_malignancy_patient_level(
    df: pd.DataFrame,
    image_col: str,
    patient_id_col: str,
    mal_label_col: str,
    mal_model,
    compute_mask_features_fn,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rows = []
    for _, r in df.iterrows():
        pid = str(r.get(patient_id_col))
        y = r.get(mal_label_col, None)
        try:
            y = int(float(y))
        except Exception:
            y = None
        paths = [_norm_path(p) for p in _to_list(r.get(image_col))]
        paths = [p for p in paths if p]
        rows.append((pid, y, paths))

    # patient-wise mean prob across images
    out_pid = []
    out_y = []
    out_p = []

    for pid, y, paths in rows:
        if not paths:
            out_pid.append(pid)
            out_y.append(y)
            out_p.append(np.nan)
            continue

        probs = []
        imgs_b = []
        feats_b = []
        for p in paths:
            # Use PIL to support non-ASCII file paths on Windows.
            try:
                img_pil = Image.open(p).convert("RGB")
            except Exception:
                continue
            # morphology mask is optional; keep original behavior for tri_root fov/msk convention
            mask_path = p.replace("fov/img", "fov/msk").replace("fov\\img", "fov\\msk")
            mask_arr = None
            try:
                if Path(mask_path).exists():
                    # read mask robustly via PIL as well
                    mask_arr = np.array(Image.open(mask_path).convert("L"))
            except Exception:
                mask_arr = None
            feats = compute_mask_features_fn(mask_arr)
            imgs_b.append(tfm(img_pil))
            feats_b.append(torch.from_numpy(np.asarray(feats, dtype=np.float32)))
            if len(imgs_b) >= batch_size:
                xb = torch.stack(imgs_b).to(device)
                fb = torch.stack(feats_b).to(device)
                logits = mal_model(xb, fb)
                pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
                probs.extend(pb)
                imgs_b, feats_b = [], []

        if imgs_b:
            xb = torch.stack(imgs_b).to(device)
            fb = torch.stack(feats_b).to(device)
            logits = mal_model(xb, fb)
            pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            probs.extend(pb)

        out_pid.append(pid)
        out_y.append(y)
        out_p.append(float(np.mean(probs)) if probs else np.nan)

    return pd.DataFrame({"patient_id": out_pid, "y_mal": out_y, "p_mal": out_p})


def _load_strat_module():
    script = Path(__file__).parent / "train_llm_enhanced_stratification_v2_optimized.py"
    spec = importlib.util.spec_from_file_location("strat_v2_optimized", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load: {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _build_strat_model_from_results(results_json: Path, ckpt: Path, device: str, llm_local_only: bool, disable_llm: bool):
    mod = _load_strat_module()
    import torch
    from transformers import AutoTokenizer, AutoModel

    with results_json.open("r", encoding="utf-8") as f:
        res = json.load(f)
    cfg = res.get("config", {})

    model_size = cfg.get("model_size", "base")
    input_size = int(cfg.get("input_size", 384))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    dropout = float(cfg.get("dropout", 0.2))
    num_prompts = int(cfg.get("num_prompts", 8))
    use_domain_embedding = bool(cfg.get("use_domain_embedding", False))
    tri_modal_num_layers = int(cfg.get("tri_modal_num_layers", 2))

    numerical_features = res.get("numerical_features", [])
    categorical_features = res.get("categorical_features", [])

    # tokenizer/llm
    tokenizer = None
    llm_model = None
    if not disable_llm:
        name = cfg.get("llm_model_name", "hfl/chinese-bert-wwm-ext")
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=bool(llm_local_only))
        llm_model = AutoModel.from_pretrained(name, local_files_only=bool(llm_local_only))

    # cat vocab size (same as training script)
    vocab_sizes = []
    if "sex" in categorical_features:
        vocab_sizes.append(2)
    if "max_diameter_category" in categorical_features:
        vocab_sizes.append(2)
    cat_vocab_size = max(vocab_sizes) if vocab_sizes else 1

    ModelCls = getattr(mod, "LLMEnhancedStratificationModelV2Optimized")
    model = ModelCls(
        num_numerical=len(numerical_features),
        num_categorical=len(categorical_features),
        cat_vocab_size=cat_vocab_size,
        img_size=input_size,
        hidden_dim=hidden_dim,
        num_prompts=num_prompts,
        llm_model=llm_model,
        freeze_llm=True,
        dropout=dropout,
        use_domain_embedding=use_domain_embedding,
        model_size=model_size,
        tri_modal_num_layers=tri_modal_num_layers,
    ).to(device)

    sd = torch.load(str(ckpt), map_location=device)
    # ckpt is usually model.state_dict()
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, tokenizer, numerical_features, categorical_features


def _predict_highrisk(
    csv_path: Path,
    image_column: str,
    model,
    tokenizer,
    numerical_scaler_pkl: Optional[Path],
    disable_llm: bool,
    device: str,
    batch_size: int,
):
    # Use dataset/feature_textualizer from strat scripts for consistency.
    mod = _load_strat_module()
    BaseDS = getattr(mod, "LLMEnhancedDatasetV2Optimized")
    FeatureTextualizer = getattr(mod, "FeatureTextualizer")

    import pickle
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # minimal transforms for eval
    tfm = transforms.Compose(
        [
            transforms.Resize((int(384), int(384))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    scaler = None
    if numerical_scaler_pkl is not None and numerical_scaler_pkl.exists():
        with numerical_scaler_pkl.open("rb") as f:
            scaler = pickle.load(f)

    # feature textualizer: it needs radiomics feature names; we can use all rad_* in csv
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    rad_cols = [c for c in df.columns if str(c).startswith("rad_")]
    ft = FeatureTextualizer(radiomics_feature_names=rad_cols, radiomics_feature_stats=None)

    ds = BaseDS(
        csv_path=str(csv_path),
        transform=tfm,
        mode="test",
        image_column=image_column,
        feature_config="age_sex_maxdiameter",
        radiomics_feature_list_path=None,  # use whatever in csv
        numerical_scaler=scaler,
        feature_textualizer=ft,
        llm_tokenizer=None if disable_llm else tokenizer,
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    all_p = []
    all_y = []
    all_pid = []

    for batch in dl:
        images = batch["image"].to(device)
        numerical = batch["numerical"].to(device)
        categorical = batch["categorical"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_id = batch.get("domain_id")
        if domain_id is not None:
            domain_id = domain_id.to(device)

        logits = model(images, numerical, categorical, input_ids, attention_mask, domain_id=domain_id)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_p.append(p)
        all_y.append(np.asarray(batch["label"], dtype=int))

    all_p = np.concatenate(all_p) if all_p else np.array([], dtype=float)
    all_y = np.concatenate(all_y) if all_y else np.array([], dtype=int)
    # patient_id is in ds.patients
    try:
        all_pid = ds.patients["patient_id"].astype(str).tolist()
    except Exception:
        all_pid = [str(i) for i in range(len(all_p))]

    return pd.DataFrame({"patient_id": all_pid, "y_high": all_y.tolist(), "p_high": all_p.tolist()})


def _compute_metrics_and_plots(
    split_name: str,
    df_tri: pd.DataFrame,
    pm: pd.Series,
    ph: pd.Series,
    out_dir: Path,
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    # build kept tri-class subset
    y_mal = pd.to_numeric(df_tri["malignancy"], errors="coerce").fillna(-1).astype(int).to_numpy()
    y_str = pd.to_numeric(df_tri["stratification"], errors="coerce").fillna(-1).astype(int).to_numpy()

    keep = (y_mal == 0) | ((y_mal == 1) & np.isin(y_str, [0, 1]))
    y3 = np.full(len(df_tri), -1, dtype=int)
    y3[y_mal == 0] = 0
    y3[(y_mal == 1) & (y_str == 0)] = 1
    y3[(y_mal == 1) & (y_str == 1)] = 2

    pmv = pm.to_numpy(dtype=float)
    phv = ph.to_numpy(dtype=float)
    pmv = np.clip(pmv, 1e-6, 1 - 1e-6)
    phv = np.clip(phv, 1e-6, 1 - 1e-6)

    p0 = 1.0 - pmv
    p2 = pmv * phv
    p1 = pmv * (1.0 - phv)
    p3 = np.stack([p0, p1, p2], axis=1)
    pred3 = np.argmax(p3, axis=1)

    y3k = y3[keep]
    p3k = p3[keep]
    pred3k = pred3[keep]

    acc3 = float(accuracy_score(y3k, pred3k)) if y3k.size else 0.0
    f1m = float(f1_score(y3k, pred3k, average="macro")) if y3k.size else 0.0
    auc3 = None
    if y3k.size and len(np.unique(y3k)) >= 2:
        try:
            auc3 = float(roc_auc_score(y3k, p3k, multi_class="ovr", average="macro"))
        except Exception:
            auc3 = None

    # malignancy auc (binary)
    auc_mal = None
    m_bin = np.isin(y_mal, [0, 1])
    if m_bin.sum() > 0 and len(np.unique(y_mal[m_bin])) >= 2:
        auc_mal = float(roc_auc_score(y_mal[m_bin], pmv[m_bin]))

    # highrisk auc on malignant
    auc_high = None
    mh = (y_mal == 1) & np.isin(y_str, [0, 1])
    if mh.sum() > 0 and len(np.unique(y_str[mh])) >= 2:
        auc_high = float(roc_auc_score(y_str[mh], phv[mh]))

    # plots
    _safe_makedirs(out_dir)
    cm3 = confusion_matrix(y3k, pred3k, labels=[0, 1, 2])
    _plot_cm(cm3, ["0(benign)", "1(mal_low)", "2(mal_high)"], out_dir / f"cm_{split_name}_triclass.png", f"{split_name} Tri-class CM")
    if auc_mal is not None:
        _plot_roc(y_mal[m_bin], pmv[m_bin], out_dir / f"roc_{split_name}_malignancy.png", f"{split_name} ROC: malignancy")
    if auc_high is not None:
        _plot_roc(y_str[mh], phv[mh], out_dir / f"roc_{split_name}_highrisk.png", f"{split_name} ROC: highrisk on malignant")

    return {
        "n_eval_kept_for_triclass": int(keep.sum()),
        "triclass_acc": acc3,
        "triclass_f1_macro": f1m,
        "triclass_auc_ovr_macro": auc3,
        "malignancy_auc": auc_mal,
        "highrisk_auc_on_malignant": auc_high,
    }


def main() -> None:
    # Windows console safety: avoid UnicodeEncodeError on non-UTF8 code pages
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tri_root", type=str, required=True, help="Dir with train.csv/val.csv/test.csv (patient-level)")
    ap.add_argument("--external_csv", type=str, default="", help="Optional external malignant-only CSV")
    ap.add_argument("--out_dir", type=str, default="result/twomodel_fusion_triclass_v1")

    ap.add_argument("--mal_ckpt", type=str, required=True, help="Malignancy classifier weights (state_dict, e.g. best.pt)")
    ap.add_argument("--mal_backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    ap.add_argument("--mal_image_col", type=str, default="image_path")

    ap.add_argument("--strat_ckpt", type=str, required=True, help="High-risk strat model weights (best_model.pth)")
    ap.add_argument("--strat_results_json", type=str, required=True, help="High-risk strat results.json (for config)")
    ap.add_argument("--strat_image_col", type=str, default="nodule_crop_path")
    ap.add_argument("--strat_scaler_pkl", type=str, default="", help="Optional numerical_scaler.pkl; default: next to ckpt")
    ap.add_argument("--llm_local_only", action="store_true")
    ap.add_argument("--disable_llm", action="store_true")

    ap.add_argument("--batch_size_mal", type=int, default=16)
    ap.add_argument("--batch_size_strat", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--calibrate_on_val", action="store_true", help="Platt-calibrate Pm and Ph on val split before fusion")
    args = ap.parse_args()

    import torch

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    _safe_makedirs(out_dir)

    tri_root = Path(args.tri_root)
    train_csv = tri_root / "train.csv"
    val_csv = tri_root / "val.csv"
    test_csv = tri_root / "test.csv"
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"tri_root must contain train.csv/val.csv/test.csv: {tri_root}")

    # load malignancy model
    mal_model, compute_mask_features_fn = _load_malignancy_model(Path(args.mal_ckpt), args.mal_backbone, device=str(device))

    # load strat model
    strat_ckpt = Path(args.strat_ckpt)
    scaler_pkl = Path(args.strat_scaler_pkl) if args.strat_scaler_pkl else (strat_ckpt.parent / "numerical_scaler.pkl")
    strat_model, tok, _, _ = _build_strat_model_from_results(
        results_json=Path(args.strat_results_json),
        ckpt=strat_ckpt,
        device=str(device),
        llm_local_only=bool(args.llm_local_only),
        disable_llm=bool(args.disable_llm),
    )

    # predict per split
    def _load_split(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    splits = {"train": _load_split(train_csv), "val": _load_split(val_csv), "test": _load_split(test_csv)}

    pm = {}
    for name, df in splits.items():
        pred = _predict_malignancy_patient_level(
            df=df,
            image_col=args.mal_image_col,
            patient_id_col="patient_id",
            mal_label_col="malignancy",
            mal_model=mal_model,
            compute_mask_features_fn=compute_mask_features_fn,
            device=str(device),
            batch_size=int(args.batch_size_mal),
        )
        pm[name] = pred.set_index("patient_id")["p_mal"]
        pred.to_csv(out_dir / f"preds_{name}_malignancy.csv", index=False, encoding="utf-8-sig")

    ph = {}
    for name, df in splits.items():
        tmp_csv = out_dir / f"__tmp_{name}.csv"
        df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
        pred = _predict_highrisk(
            csv_path=tmp_csv,
            image_column=args.strat_image_col,
            model=strat_model,
            tokenizer=tok,
            numerical_scaler_pkl=scaler_pkl if scaler_pkl.exists() else None,
            disable_llm=bool(args.disable_llm),
            device=str(device),
            batch_size=int(args.batch_size_strat),
        )
        ph[name] = pred.set_index("patient_id")["p_high"]
        pred.to_csv(out_dir / f"preds_{name}_highrisk.csv", index=False, encoding="utf-8-sig")
        try:
            tmp_csv.unlink()
        except Exception:
            pass

    # optional calibration (on val)
    cal_m = None
    cal_h = None
    if args.calibrate_on_val:
        dfv = splits["val"].copy()
        y_m = pd.to_numeric(dfv["malignancy"], errors="coerce").fillna(-1).astype(int)
        m_ok = np.isin(y_m.to_numpy(), [0, 1])
        if m_ok.sum() > 0:
            p = pm["val"].reindex(dfv["patient_id"].astype(str)).to_numpy(dtype=float)
            cal_m = _fit_platt(y_m.to_numpy()[m_ok], p[m_ok])
        y_s = pd.to_numeric(dfv["stratification"], errors="coerce").fillna(-1).astype(int)
        mh = (y_m == 1) & np.isin(y_s, [0, 1])
        if mh.sum() > 0:
            p = ph["val"].reindex(dfv["patient_id"].astype(str)).to_numpy(dtype=float)
            cal_h = _fit_platt(y_s.to_numpy()[mh.to_numpy()], p[mh.to_numpy()])

    def _apply_cal(p: pd.Series, cal: Optional[PlattCalibrator]) -> pd.Series:
        if cal is None:
            return p
        arr = p.to_numpy(dtype=float)
        return pd.Series(cal.apply(arr), index=p.index)

    # metrics
    metrics = {}
    for name, df in splits.items():
        pm_s = _apply_cal(pm[name].reindex(df["patient_id"].astype(str)), cal_m).fillna(0.5)
        ph_s = _apply_cal(ph[name].reindex(df["patient_id"].astype(str)), cal_h).fillna(0.5)
        metrics[name] = _compute_metrics_and_plots(name, df, pm_s, ph_s, out_dir)

    # external highrisk only
    external_metrics = None
    if args.external_csv:
        ext_csv = Path(args.external_csv)
        df_ext = pd.read_csv(ext_csv, encoding="utf-8-sig", low_memory=False)
        tmp_csv = out_dir / "__tmp_external.csv"
        df_ext.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
        pred_ext = _predict_highrisk(
            csv_path=tmp_csv,
            image_column=args.strat_image_col,
            model=strat_model,
            tokenizer=tok,
            numerical_scaler_pkl=scaler_pkl if scaler_pkl.exists() else None,
            disable_llm=bool(args.disable_llm),
            device=str(device),
            batch_size=int(args.batch_size_strat),
        )
        try:
            tmp_csv.unlink()
        except Exception:
            pass

        # compute AUC on malignant (external assumed malignant-only)
        from sklearn.metrics import roc_auc_score

        y = pred_ext["y_high"].to_numpy(dtype=int)
        p = pred_ext["p_high"].to_numpy(dtype=float)
        auc = None
        if len(np.unique(y)) >= 2:
            auc = float(roc_auc_score(y, p))
            _plot_roc(y, p, out_dir / "roc_external_highrisk.png", "External ROC: highrisk on malignant")
        external_metrics = {
            "n_eval_malignant_with_strat": int(len(y)),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
            "highrisk_auc_on_malignant": auc,
        }
        pred_ext.to_csv(out_dir / "preds_external_highrisk.csv", index=False, encoding="utf-8-sig")

    out = {
        "tri_root": str(tri_root).replace("\\", "/"),
        "mal_ckpt": str(Path(args.mal_ckpt)).replace("\\", "/"),
        "strat_ckpt": str(strat_ckpt).replace("\\", "/"),
        "strat_results_json": str(Path(args.strat_results_json)).replace("\\", "/"),
        "calibrate_on_val": bool(args.calibrate_on_val),
        "calibrator_malignancy": None if cal_m is None else {"a": cal_m.a, "b": cal_m.b},
        "calibrator_highrisk": None if cal_h is None else {"a": cal_h.a, "b": cal_h.b},
        "val": metrics.get("val"),
        "test": metrics.get("test"),
        "external": external_metrics,
    }
    (out_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", str(out_dir / "results.json"))


if __name__ == "__main__":
    # torch is used in decorators; import here to avoid hard dependency at file import time
    import torch  # noqa: F401

    main()

