@echo off
chcp 65001 > nul
title AgriVision AI - Web Demo
cd /d "%~dp0"

rem Chọn Python: ưu tiên venv_gpu, sau đó venv, cuối cùng là python hệ thống
set "PY=python"
if exist ".\venv\Scripts\python.exe" set "PY=.\venv\Scripts\python.exe"
if exist ".\venv_gpu\Scripts\python.exe" set "PY=.\venv_gpu\Scripts\python.exe"

rem Mặc định dùng mô hình huấn luyện trên cách chia theo nhóm (models/tier3_resnet50_group.pth)
if not defined PV_SPLIT set "PV_SPLIT=group"

echo ===================================================
echo   KHOI DONG HE THONG CHUAN DOAN BENH CAY TRONG
echo ===================================================
echo -^> Python: %PY%  (PV_SPLIT=%PV_SPLIT%)
echo -^> Dang nap model ResNet50 (GPU neu co, khong thi CPU)...
echo -^> Mo trinh duyet tai: http://localhost:5000
echo ===================================================
"%PY%" -m demo.app
pause
