#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run two-model fusion inference on data/cexu_test.

Inputs:
- cexu_test root: each subfolder is one case, contains PNG images
- info.xlsx: case metadata; Excel column T maps to folder name (confirmed as df.columns[19])

Models:
- malignancy classifier: result/malignancy_cls_*/best.pt (state_dict from train_malignancy_classifier.py)
- highrisk strat model: result/llm_enhanced_stratification_*/best_model.pth + results.json (v2_optimized)

Outputs:
- out_xlsx: merged info + predictions
- optional intermediate CSV used for highrisk prediction
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
                return _to_list(ast.literal_eval(s))
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def _norm_path(s: str) -> str:
    return str(s).strip().strip("'\"").replace("\\", "/")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _safe_int(x: Any) -> Optional[int]:
    v = _safe_float(x)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_mask_json_for_image(image_path: Path) -> Optional[Path]:
    """
    For cexu_test, each PNG often has a sibling JSON (LabelMe-like) with polygon points.
    Prefer <stem>.json.
    """
    p = image_path.with_suffix(".json")
    if p.exists():
        return p
    return None


def _rasterize_mask_from_labelme_json(json_path: Path) -> Optional[np.ndarray]:
    """
    Rasterize LabelMe-style polygon shapes into a binary mask.
    Returns uint8 mask (0/255) or None if failed.
    """
    try:
        import cv2

        d = json.loads(json_path.read_text(encoding="utf-8"))
        w = int(d.get("imageWidth") or 0)
        h = int(d.get("imageHeight") or 0)
        if w <= 0 or h <= 0:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = d.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            return None
        polys = []
        for s in shapes:
            if not isinstance(s, dict):
                continue
            if str(s.get("shape_type", "")).lower() != "polygon":
                continue
            pts = s.get("points", [])
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            arr = np.array([[float(x), float(y)] for x, y in pts], dtype=np.float32)
            polys.append(arr.reshape(-1, 1, 2))
        if not polys:
            return None
        cv2.fillPoly(mask, [p.astype(np.int32) for p in polys], 255)
        return mask
    except Exception:
        return None


def _load_malignancy_model(ckpt: Path, backbone: str, device: str):
    import torch

    sd_raw = torch.load(str(ckpt), map_location=device)
    # support checkpoint wrappers
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
        sd = sd_raw["state_dict"]
    elif isinstance(sd_raw, dict) and "model_state_dict" in sd_raw and isinstance(sd_raw["model_state_dict"], dict):
        sd = sd_raw["model_state_dict"]
    else:
        sd = sd_raw

    keys = list(sd.keys()) if isinstance(sd, dict) else []
    # Auto-pick the right model definition by key prefix.
    # - v1 script: cnn.* / cnn_head.* / feat_head.*
    # - v3 script: backbone.* / morphology_fc.* (your malignancy_cls_v3 checkpoint)
    use_v3 = any(k.startswith("backbone.") for k in keys) or any(k.startswith("morphology_fc.") for k in keys)

    if use_v3:
        script = Path(__file__).parent / "train_malignancy_classifier_v3.py"
        mod = _load_module(script, "train_malignancy_classifier_v3")
    else:
        script = Path(__file__).parent / "train_malignancy_classifier.py"
        mod = _load_module(script, "train_malignancy_classifier")

    CNNWithMorphology = getattr(mod, "CNNWithMorphology")
    compute_mask_features = getattr(mod, "compute_mask_features")

    model = CNNWithMorphology(backbone=backbone, pretrained=False).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        # last resort: allow non-strict load (shouldn't happen when we picked the right script)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, compute_mask_features


def _load_strat_model_from_results(results_json: Path, ckpt: Path, device: str, llm_local_only: bool):
    import torch
    from transformers import AutoTokenizer, AutoModel

    script = Path(__file__).parent / "train_llm_enhanced_stratification_v2_optimized.py"
    mod = _load_module(script, "train_llm_enhanced_stratification_v2_optimized")

    with results_json.open("r", encoding="utf-8") as f:
        res = json.load(f)
    cfg = res.get("config", {})

    model_size = cfg.get("model_size", "base")
    input_size = int(cfg.get("input_size", 384))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    dropout = float(cfg.get("dropout", 0.3))
    num_prompts = int(cfg.get("num_prompts", 8))
    use_domain_embedding = bool(cfg.get("use_domain_embedding", False))
    tri_modal_num_layers = int(cfg.get("tri_modal_num_layers", 2))

    numerical_features = res.get("numerical_features", [])
    categorical_features = res.get("categorical_features", [])
    radiomics_feature_list_path = cfg.get("radiomics_feature_list", None)

    # tokenizer/llm
    name = cfg.get("llm_model_name", "hfl/chinese-bert-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=bool(llm_local_only))
    llm_model = AutoModel.from_pretrained(name, local_files_only=bool(llm_local_only))

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
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, tokenizer, radiomics_feature_list_path


def _extract_radiomics_one(image_path: Path) -> Dict[str, float]:
    script = Path(__file__).parent / "extract_radiomics_features.py"
    mod = _load_module(script, "extract_radiomics_features")
    extract_fn = getattr(mod, "extract_radiomics_features")
    d = extract_fn(image_path, mask_path=None, extractor=None)
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            vv = float(v)
            if np.isfinite(vv):
                out[str(k)] = vv
        except Exception:
            continue
    return out


def _aggregate_radiomics(feat_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float, float]]:
    keys = sorted(set().union(*[set(d.keys()) for d in feat_list])) if feat_list else []
    out: Dict[str, Tuple[float, float, float]] = {}
    for k in keys:
        vals = [d.get(k, np.nan) for d in feat_list]
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[k] = (0.0, 0.0, 0.0)
        else:
            out[k] = (float(np.mean(arr)), float(np.max(arr)), float(np.min(arr)))
    return out


def _predict_malignancy_case(image_paths: List[Path], mal_model, compute_mask_features_fn, device: str, batch_size: int) -> float:
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

    probs: List[float] = []
    imgs_b = []
    feats_b = []
    for p in image_paths:
        # Use PIL to support non-ASCII (Chinese) file paths on Windows.
        try:
            img_pil = Image.open(p).convert("RGB")
        except Exception:
            continue
        # Prefer polygon mask from sibling JSON if available; else fallback to zeros.
        mjson = _find_mask_json_for_image(p)
        mask_arr = _rasterize_mask_from_labelme_json(mjson) if mjson else None
        feats = compute_mask_features_fn(mask_arr)
        imgs_b.append(tfm(img_pil))
        feats_b.append(torch.from_numpy(np.asarray(feats, dtype=np.float32)))
        if len(imgs_b) >= batch_size:
            xb = torch.stack(imgs_b).to(device)
            fb = torch.stack(feats_b).to(device)
            logits = mal_model(xb, fb)
            pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            probs.extend([float(x) for x in pb])
            imgs_b, feats_b = [], []

    if imgs_b:
        xb = torch.stack(imgs_b).to(device)
        fb = torch.stack(feats_b).to(device)
        logits = mal_model(xb, fb)
        pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
        probs.extend([float(x) for x in pb])

    if not probs:
        return 0.5
    return float(np.mean(probs))


def _predict_highrisk_from_csv(
    tmp_csv: Path,
    image_column: str,
    strat_model,
    tokenizer,
    device: str,
    batch_size: int,
    radiomics_feature_list_path: Optional[str],
) -> pd.DataFrame:
    # build dataset from v2_optimized script to be consistent
    mod = _load_module(Path(__file__).parent / "train_llm_enhanced_stratification_v2_optimized.py", "strat_v2_optimized")
    BaseDS = getattr(mod, "LLMEnhancedDatasetV2Optimized")
    FeatureTextualizer = getattr(mod, "FeatureTextualizer")

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    df = pd.read_csv(tmp_csv, encoding="utf-8-sig", low_memory=False)
    rad_cols = [c for c in df.columns if str(c).startswith("rad_")]
    ft = FeatureTextualizer(radiomics_feature_names=rad_cols, radiomics_feature_stats=None)

    tfm = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = BaseDS(
        csv_path=str(tmp_csv),
        transform=tfm,
        mode="test",
        image_column=image_column,
        feature_config="age_sex_maxdiameter",
        radiomics_feature_list_path=radiomics_feature_list_path,
        numerical_scaler=None,
        feature_textualizer=ft,
        llm_tokenizer=tokenizer,
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    ps = []
    pids = ds.patients["patient_id"].astype(str).tolist()
    for batch in dl:
        images = batch["image"].to(device)
        numerical = batch["numerical"].to(device)
        categorical = batch["categorical"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_id = batch.get("domain_id")
        if domain_id is not None:
            domain_id = domain_id.to(device)
        logits = strat_model(images, numerical, categorical, input_ids, attention_mask, domain_id=domain_id)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
        ps.extend([float(x) for x in p])

    return pd.DataFrame({"case_id": pids, "p_high": ps})


def main() -> None:
    # Windows console safety: avoid UnicodeEncodeError on non-UTF8 code pages
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cexu_root", type=str, default="data/cexu_test")
    ap.add_argument("--info_xlsx", type=str, default="data/cexu_test/info.xlsx")
    ap.add_argument("--id_col_name", type=str, default="", help="If provided, use this column as folder id; else use Excel column T (index 19)")

    ap.add_argument("--out_dir", type=str, default="result/cexu_test_fusion_pred_v1")
    ap.add_argument("--out_xlsx", type=str, default="")

    ap.add_argument("--mal_ckpt", type=str, required=True)
    ap.add_argument("--mal_backbone", type=str, default="efficientnet_b0")
    ap.add_argument("--strat_ckpt", type=str, required=True)
    ap.add_argument("--strat_results_json", type=str, required=True)
    ap.add_argument("--llm_local_only", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size_mal", type=int, default=16)
    ap.add_argument("--batch_size_strat", type=int, default=16)
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

    import torch

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else (out_dir / "cexu_test_predictions.xlsx")

    cexu_root = Path(args.cexu_root)
    info_xlsx = Path(args.info_xlsx)
    if not info_xlsx.exists():
        raise FileNotFoundError(f"info_xlsx not found: {info_xlsx}")

    # Read info.xlsx (requires openpyxl)
    try:
        info_df = pd.read_excel(info_xlsx)
    except Exception as e:
        raise RuntimeError(f"Failed to read xlsx (need openpyxl). Error: {e}")

    # Determine case id column (Excel column T = index 19)
    if args.id_col_name and args.id_col_name in info_df.columns:
        id_col = args.id_col_name
    else:
        if len(info_df.columns) < 20:
            raise RuntimeError("info.xlsx has < 20 columns; cannot use column T")
        id_col = info_df.columns[19]

    # Map metadata columns -> model features
    col_sex = "性别" if "性别" in info_df.columns else ("sex" if "sex" in info_df.columns else None)
    col_age = "年龄" if "年龄" in info_df.columns else ("age" if "age" in info_df.columns else None)
    col_maxd = "穿刺结节最大径mm" if "穿刺结节最大径mm" in info_df.columns else None

    # Load models
    mal_model, compute_mask_features_fn = _load_malignancy_model(Path(args.mal_ckpt), args.mal_backbone, device=str(device))
    strat_model, tokenizer, rad_list_path = _load_strat_model_from_results(
        results_json=Path(args.strat_results_json),
        ckpt=Path(args.strat_ckpt),
        device=str(device),
        llm_local_only=bool(args.llm_local_only),
    )

    # Build per-case table for highrisk model (patient-level CSV)
    case_rows = []
    preds_rows = []

    ids = info_df[id_col].astype(str).tolist()
    if args.max_cases is not None:
        ids = ids[: int(args.max_cases)]
        info_df = info_df.iloc[: int(args.max_cases)].copy()

    # Pre-load radiomics extractor module once
    rad_mod = _load_module(Path(__file__).parent / "extract_radiomics_features.py", "extract_radiomics_features")
    extract_fn = getattr(rad_mod, "extract_radiomics_features")

    for i, row in info_df.iterrows():
        case_id = str(row.get(id_col, "")).strip()
        if not case_id or case_id.lower() == "nan":
            continue
        folder = cexu_root / case_id
        imgs = []
        if folder.exists() and folder.is_dir():
            imgs = sorted([p for p in folder.glob("*.png") if p.is_file()])
        if not imgs:
            preds_rows.append({"case_id": case_id, "status": "no_images", "num_images": 0})
            continue

        # radiomics per image
        feat_list = []
        for p in imgs:
            mjson = _find_mask_json_for_image(p)
            try:
                d = extract_fn(p, mask_path=mjson, extractor=None)
            except Exception:
                continue
            dd = {}
            for k, v in d.items():
                if not str(k).startswith("rad_"):
                    continue
                try:
                    vv = float(v)
                    if np.isfinite(vv):
                        dd[str(k)] = vv
                except Exception:
                    continue
            if dd:
                feat_list.append(dd)
        agg = _aggregate_radiomics(feat_list)

        sex = row.get(col_sex, None) if col_sex else None
        age = row.get(col_age, None) if col_age else None
        maxd = row.get(col_maxd, None) if col_maxd else None
        maxd_f = _safe_float(maxd)
        maxd_cat = 1 if (maxd_f is not None and maxd_f > 10.0) else 0

        # store row for strat model csv
        out = {
            "patient_id": case_id,
            "source": "cexu_test",
            "stratification": 0,  # dummy label for dataset
            "sex": sex,
            "age_years": _safe_float(age),
            "max_diameter_mm": maxd_f,
            "max_diameter_category": maxd_cat,
            "nodule_crop_path": json.dumps([_norm_path(str(p)) for p in imgs], ensure_ascii=False),
        }
        for k, (m, mx, mn) in agg.items():
            out[f"{k}_mean"] = m
            out[f"{k}_max"] = mx
            out[f"{k}_min"] = mn
        case_rows.append(out)

        # malignancy prediction now (on same images)
        p_mal = _predict_malignancy_case(imgs, mal_model, compute_mask_features_fn, device=str(device), batch_size=int(args.batch_size_mal))
        preds_rows.append({"case_id": case_id, "status": "ok", "num_images": int(len(imgs)), "p_mal": float(p_mal)})

    tmp_csv = out_dir / "__tmp_cexu_patientlevel.csv"
    df_case = pd.DataFrame(case_rows)
    # Ensure required radiomics columns exist (avoid KeyError in dataset)
    if rad_list_path:
        rad_list_file = Path(str(rad_list_path))
        if rad_list_file.exists():
            req = []
            for line in rad_list_file.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    req.append(s)
            for c in req:
                if c not in df_case.columns:
                    df_case[c] = 0.0
    df_case.to_csv(tmp_csv, index=False, encoding="utf-8-sig")

    # highrisk probabilities for cases that had images
    df_high = _predict_highrisk_from_csv(
        tmp_csv,
        image_column="nodule_crop_path",
        strat_model=strat_model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=int(args.batch_size_strat),
        radiomics_feature_list_path=rad_list_path,
    )

    # merge
    df_pred = pd.DataFrame(preds_rows)
    df_pred = df_pred.merge(df_high, on="case_id", how="left")
    df_pred["p_mal"] = pd.to_numeric(df_pred.get("p_mal"), errors="coerce").fillna(0.5)
    df_pred["p_high"] = pd.to_numeric(df_pred.get("p_high"), errors="coerce").fillna(0.5)

    pm = df_pred["p_mal"].to_numpy(dtype=float)
    ph = df_pred["p_high"].to_numpy(dtype=float)
    p0 = 1.0 - pm
    p2 = pm * ph
    p1 = pm * (1.0 - ph)
    pred = np.argmax(np.stack([p0, p1, p2], axis=1), axis=1)

    df_pred["p0_benign"] = p0
    df_pred["p1_mal_low"] = p1
    df_pred["p2_mal_high"] = p2
    df_pred["pred_class"] = pred.astype(int)
    df_pred["pred_name"] = df_pred["pred_class"].map({0: "benign", 1: "mal_low", 2: "mal_high"})

    # merge back to info
    info_df["_case_id_"] = info_df[id_col].astype(str).str.strip()
    out_df = info_df.merge(df_pred, left_on="_case_id_", right_on="case_id", how="left")

    # write excel
    try:
        out_df.to_excel(out_xlsx, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write xlsx (need openpyxl). Error: {e}")

    # also write csv
    out_df.to_csv(out_dir / "cexu_test_predictions.csv", index=False, encoding="utf-8-sig")
    print("Wrote:", str(out_xlsx).replace("\\", "/"))


if __name__ == "__main__":
    main()

