# app.py
import streamlit as st
import requests
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
import plotly.express as px
import time

API_BASE = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_BASE}/predict"
SPEECH_ENDPOINT = f"{API_BASE}/speech"

st.set_page_config(page_title="Lớp học AI - Theo dõi lời nói độc hại", layout="wide")
st.title("🎧 Lớp học AI — Theo dõi lời nói độc hại (Tiếng Việt)")

# Session state store for recent results
if "history" not in st.session_state:
    st.session_state.history = []  # list of dict: {ts, text, label, score}

col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio("Chọn chế độ", ("Nhập văn bản", "Ghi âm"), horizontal=True)

    if mode == "Nhập văn bản":
        text = st.text_area("Nhập văn bản tiếng Việt:", height=150)
        if st.button("Phân tích văn bản"):
            if not text.strip():
                st.warning("Vui lòng nhập nội dung.")
            else:
                try:
                    with st.spinner("Đang gửi..."):
                        r = requests.post(PREDICT_ENDPOINT, json={"text": text}, timeout=20)
                        r.raise_for_status()
                        data = r.json()
                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}")
                else:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.insert(0, {"ts": ts, "text": data["text"], "label": data["label"], "score": float(data["score"])})
                    if data["label"] == "Độc hại":
                        st.error(f"🚫 Độc hại — độ tin cậy {data['score']:.2%}")
                    else:
                        st.success(f"✅ Bình thường — độ tin cậy {data['score']:.2%}")

    else:
        st.write("Nhấn **Bắt đầu ghi âm** để ghi 3–10 giây giọng nói, hệ thống sẽ gửi để nhận dạng và phân loại.")
        duration = st.slider("Thời lượng ghi (giây):", min_value=2, max_value=10, value=5)
        if st.button("Bắt đầu ghi âm"):
            fs = 16000
            st.info("🔴 Đang ghi âm... Nói ngay bây giờ.")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
            sd.wait()
            st.success("⏺️ Ghi âm xong.")
            # save temp wav
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, recording, fs)
            st.audio(tmp.name)
            # send file
            try:
                with st.spinner("Đang gửi âm thanh tới API..."):
                    with open(tmp.name, "rb") as f:
                        files = {"file": (tmp.name, f, "audio/wav")}
                        r = requests.post(SPEECH_ENDPOINT, files=files, timeout=60)
                        r.raise_for_status()
                        data = r.json()
            except Exception as e:
                st.error(f"Lỗi khi gọi API: {e}")
            else:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.insert(0, {"ts": ts, "text": data.get("text",""), "label": data.get("label",""), "score": float(data.get("score",0.0))})
                if data.get("label") == "Độc hại":
                    st.error(f"🚫 Độc hại — độ tin cậy {data['score']:.2%}")
                else:
                    st.success(f"✅ Bình thường — độ tin cậy {data['score']:.2%}")

with col2:
    st.header("Chỉ số nhanh")
    df_hist = pd.DataFrame(st.session_state.history)
    total = len(df_hist)
    toxic = df_hist[df_hist["label"] == "Độc hại"].shape[0] if total>0 else 0
    rate = (toxic / total)*100 if total>0 else 0.0
    st.metric("Tổng mẫu", total)
    st.metric("Số độc hại", toxic)
    st.metric("Tỷ lệ độc hại", f"{rate:.1f}%")

    st.markdown("### Lịch sử gần đây")
    if total > 0:
        st.dataframe(df_hist[["ts","text","label","score"]].head(20), use_container_width=True)
        # biểu đồ cột: số độc hại / không
        fig = px.pie(names=["Độc hại","Bình thường"], values=[toxic, max(0,total-toxic)], title="Tỷ lệ", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Chưa có mẫu nào.")
