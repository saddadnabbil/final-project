@echo off
echo ============================================
echo RUNNING REAL DATA EVALUATION
echo ============================================
cd /d C:\laragon\www\skripsi\test-huggingface
backend\.venv\Scripts\python.exe scripts\evaluate_real_data.py --samples 5 --seed 42
echo.
echo ============================================
echo EVALUATION COMPLETE
echo ============================================
pause
