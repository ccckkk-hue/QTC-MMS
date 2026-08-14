@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Scheme-1: Two-model fusion (eval-only)
echo - malignancy model: train_malignancy_classifier.py best.pt
echo - highrisk model: train_llm_enhanced_stratification_v2_optimized.py best_model.pth
echo Outputs: ROC + CM + results.json
echo ========================================
echo.

REM --------- You need to set these paths ---------
set TRI_ROOT=data\Mask\MULTITASK_BENIGN_MALIGNANT_STRATIFICATION_routeA_roi_radiomics_v1
set EXTERNAL_CSV=data\Mask\MALIGNANT_STRATIFICATION_PATIENT_LEVEL\external_shaoxing1_preprocessed_roi_radiomics_fixed_categorized.csv

REM malignancy classifier (your latest)
set MAL_CKPT=result\malignancy_cls_v3\best.pt

REM high-risk LLM stratification model (your latest)
set STRAT_DIR=result\llm_enhanced_stratification_v2_optimized_beimo_push_v1
set STRAT_CKPT=%STRAT_DIR%\best_model.pth
set STRAT_RESULTS=%STRAT_DIR%\results.json

REM output
set OUT_DIR=result\scheme_fusion_eval_only_v1
mkdir "%OUT_DIR%" >nul 2>&1

REM --------- Preflight checks ---------
if not exist "%TRI_ROOT%\train.csv" (
  echo [FAIL] TRI_ROOT missing train.csv: %TRI_ROOT%
  pause
  exit /b 1
)
if not exist "%TRI_ROOT%\val.csv" (
  echo [FAIL] TRI_ROOT missing val.csv: %TRI_ROOT%
  pause
  exit /b 1
)
if not exist "%TRI_ROOT%\test.csv" (
  echo [FAIL] TRI_ROOT missing test.csv: %TRI_ROOT%
  pause
  exit /b 1
)
if not exist "%MAL_CKPT%" (
  echo [FAIL] MAL_CKPT not found: %MAL_CKPT%
  pause
  exit /b 1
)
if not exist "%STRAT_CKPT%" (
  echo [FAIL] STRAT_CKPT not found: %STRAT_CKPT%
  pause
  exit /b 1
)
if not exist "%STRAT_RESULTS%" (
  echo [FAIL] STRAT_RESULTS not found: %STRAT_RESULTS%
  pause
  exit /b 1
)

python scripts\eval_triclass_fusion_two_models_v1.py ^
  --tri_root "%TRI_ROOT%" ^
  --external_csv "%EXTERNAL_CSV%" ^
  --out_dir "%OUT_DIR%" ^
  --mal_ckpt "%MAL_CKPT%" ^
  --mal_backbone efficientnet_b0 ^
  --mal_image_col "image_path" ^
  --strat_ckpt "%STRAT_CKPT%" ^
  --strat_results_json "%STRAT_RESULTS%" ^
  --strat_image_col "nodule_crop_path" ^
  --llm_local_only ^
  --calibrate_on_val ^
  --batch_size_mal 16 ^
  --batch_size_strat 16

if errorlevel 1 (
  echo [FAIL] fusion eval failed.
  pause
  exit /b 1
)

echo.
echo Done.
echo - OUT: %OUT_DIR%
echo - results.json: %OUT_DIR%\results.json
pause

