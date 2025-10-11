HƯỚNG DẪN KHỞI ĐỘNG (Windows, PowerShell)

1) Tạo và kích hoạt virtualenv:
   python -m venv venv
   .\venv\Scripts\Activate.ps1

2) Cài dependencies:
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

3) Chạy API:
   uvicorn api:app --reload


4) Mở giao diện:
   streamlit run app.py

Ghi chú:
- Lần chạy đầu tiên sẽ tải các mô hình từ Hugging Face (khoảng vài trăm MB).
- Nếu gặp lỗi liên quan đến sounddevice trên Windows, cài PortAudio / wheel tương ứng.
