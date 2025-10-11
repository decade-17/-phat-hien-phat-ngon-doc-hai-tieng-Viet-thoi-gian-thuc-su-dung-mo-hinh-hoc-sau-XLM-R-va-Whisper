@echo off
title 🚀 Toxicity Classifier - Auto Launcher
color 0A

echo ====================================================
echo   🎧 DANG KHOI DONG TOXICITY CLASSIFIER (TIENG VIET)
echo ====================================================

:: Bật môi trường ảo
call venv\Scripts\activate

:: Mở cửa sổ riêng cho backend
start "FastAPI Backend" cmd /k "uvicorn api:app --reload --host 127.0.0.1 --port 8000"

:: Chờ 5 giây cho API khởi động
echo 🔹 Dang doi API khoi dong...
timeout /t 5 >nul

:: Mở frontend (Streamlit)
echo 🔹 Dang mo giao dien web...
start "Streamlit Frontend" cmd /k "streamlit run app.py"

echo ====================================================
echo ✅ Tat ca da san sang! Mo trinh duyet de su dung:
echo     👉 http://localhost:8501
echo ====================================================

pause
