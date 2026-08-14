@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Scheme-2: Train two models + fuse to tri-class
echo - Train malignancy classifier (FOV + mask morphology)
echo - Train highrisk LLM strat model (ROI + tab + text)
echo - Fuse + output ROC/CM/results.json
echo ========================================
echo.

REM -------- Inputs --------
set BM_CSV=data\Mask\clean\classification_list.csv
set TRI_ROOT=data\Mask\MULTITASK_BENIGN_MALIGNANT_STRATIFICATION_routeA_roi_radiomics_v1
set EXTERNAL_CSV=data\Mask\MALIGNANT_STRATIFICATION_PATIENT_LEVEL\external_shaoxing1_preprocessed_roi_radiomics_fixed_categorized.csv

REM -------- Outputs --------
set MAL_DIR=result\malignancy_cls_scheme2_v1
set STRAT_DIR=result\strat_llm_scheme2_v1
set FUSION_DIR=result\scheme_fusion_train_and_eval_v1

mkdir "%MAL_DIR%" >nul 2>&1
mkdir "%STRAT_DIR%" >nul 2>&1
mkdir "%FUSION_DIR%" >nul 2>&1

echo.
echo [A] Train malignancy classifier...
if exist "%MAL_DIR%\best.pt" (
  echo [SKIP] malignancy ckpt exists: %MAL_DIR%\best.pt
) else (
  python scripts\train_malignancy_classifier.py ^
    --csv "%BM_CSV%" ^
    --outdir "%MAL_DIR%" ^
    --backbone efficientnet_b0 ^
    --epochs 12 ^
    --batch 16 ^
    --lr 1e-4
  if errorlevel 1 (
    echo [FAIL] malignancy training failed.
    pause
    exit /b 1
  )
)

echo.
echo [B] Train highrisk LLM strat model...
if exist "%STRAT_DIR%\best_model.pth" (
  echo [SKIP] strat ckpt exists: %STRAT_DIR%\best_model.pth
) else (
  python scripts\train_llm_enhanced_stratification_v2_optimized.py ^
    --train_csv "data/Mask/MALIGNANT_STRATIFICATION_PATIENT_LEVEL/train_preprocessed_roi_radiomics_fixed_categorized.csv" ^
    --val_csv "data/Mask/MALIGNANT_STRATIFICATION_PATIENT_LEVEL/val_preprocessed_roi_radiomics_fixed_categorized.csv" ^
    --test_csv "data/Mask/MALIGNANT_STRATIFICATION_PATIENT_LEVEL/test_preprocessed_roi_radiomics_fixed_categorized.csv" ^
    --external_csv "%EXTERNAL_CSV%" ^
    --output_dir "%STRAT_DIR%" ^
    --image_column "nodule_crop_path" ^
    --feature_config "age_sex_maxdiameter" ^
    --batch_size 16 ^
    --epochs 120 ^
    --lr 8e-5 ^
    --image_lr_mult 0.2 ^
    --warmup_epochs 10 ^
    --freeze_image_epochs 3 ^
    --weight_decay 2e-05 ^
    --dropout 0.2 ^
    --label_smoothing 0.02 ^
    --input_size 384 ^
    --patience 25 ^
    --use_focal ^
    --radiomics_feature_list "data/Mask/MALIGNANT_STRATIFICATION_PATIENT_LEVEL/radiomics_feature_analysis_selected_features_top30.txt" ^
    --num_prompts 8 ^
    --llm_model_name "hfl/chinese-bert-wwm-ext" ^
    --freeze_llm ^
    --llm_local_only
  if errorlevel 1 (
    echo [FAIL] strat training failed.
    pause
    exit /b 1
  )
)

echo.
echo [C] Fuse and evaluate on tri-class splits...
python scripts\eval_triclass_fusion_two_models_v1.py ^
  --tri_root "%TRI_ROOT%" ^
  --external_csv "%EXTERNAL_CSV%" ^
  --out_dir "%FUSION_DIR%" ^
  --mal_ckpt "%MAL_DIR%\best.pt" ^
  --mal_backbone efficientnet_b0 ^
  --mal_image_col "image_path" ^
  --strat_ckpt "%STRAT_DIR%\best_model.pth" ^
  --strat_results_json "%STRAT_DIR%\results.json" ^
  --strat_image_col "nodule_crop_path" ^
  --llm_local_only ^
  --calibrate_on_val

if errorlevel 1 (
  echo [FAIL] fusion eval failed.
  pause
  exit /b 1
)

echo.
echo Done.
echo - MAL: %MAL_DIR%
echo - STRAT: %STRAT_DIR%
echo - FUSION: %FUSION_DIR%
echo - results.json: %FUSION_DIR%\results.json
pause

