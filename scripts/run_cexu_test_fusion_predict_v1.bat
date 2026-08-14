@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo cexu_test: Two-model fusion prediction
echo - read data\cexu_test\info.xlsx
echo - column T (图表中的组名2) maps to folder name
echo - output: result\cexu_test_fusion_pred_v1\cexu_test_predictions.xlsx
echo ========================================
echo.

set CEXU_ROOT=data\cexu_test
set INFO_XLSX=data\cexu_test\info.xlsx

REM best malignancy classifier
set MAL_CKPT=result\malignancy_cls_v3\best.pt

REM best highrisk stratification model
set STRAT_DIR=result\llm_enhanced_stratification_v2_optimized_beimo_push_v1
set STRAT_CKPT=%STRAT_DIR%\best_model.pth
set STRAT_RESULTS=%STRAT_DIR%\results.json

set OUT_DIR=result\cexu_test_fusion_pred_v1
mkdir "%OUT_DIR%" >nul 2>&1

REM --------- Preflight checks ---------
if not exist "%INFO_XLSX%" (
  echo [FAIL] missing: %INFO_XLSX%
  pause
  exit /b 1
)
if not exist "%MAL_CKPT%" (
  echo [FAIL] missing: %MAL_CKPT%
  pause
  exit /b 1
)
if not exist "%STRAT_CKPT%" (
  echo [FAIL] missing: %STRAT_CKPT%
  pause
  exit /b 1
)
if not exist "%STRAT_RESULTS%" (
  echo [FAIL] missing: %STRAT_RESULTS%
  pause
  exit /b 1
)

python scripts\predict_cexu_test_fusion_v1.py ^
  --cexu_root "%CEXU_ROOT%" ^
  --info_xlsx "%INFO_XLSX%" ^
  --out_dir "%OUT_DIR%" ^
  --mal_ckpt "%MAL_CKPT%" ^
  --mal_backbone efficientnet_b0 ^
  --strat_ckpt "%STRAT_CKPT%" ^
  --strat_results_json "%STRAT_RESULTS%" ^
  --llm_local_only ^
  --device cuda ^
  --batch_size_mal 16 ^
  --batch_size_strat 16

if errorlevel 1 (
  echo [FAIL] cexu_test prediction failed.
  pause
  exit /b 1
)

echo.
echo Done.
echo - OUT: %OUT_DIR%
echo - XLSX: %OUT_DIR%\cexu_test_predictions.xlsx
echo - CSV : %OUT_DIR%\cexu_test_predictions.csv
pause

