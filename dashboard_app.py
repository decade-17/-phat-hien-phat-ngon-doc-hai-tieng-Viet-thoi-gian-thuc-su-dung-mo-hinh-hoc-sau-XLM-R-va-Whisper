# dashboard_app.py
import streamlit as st
import requests
import pandas as pd
import time

API_STATS = "http://127.0.0.1:8000/stats"
API_RECENT = "http://127.0.0.1:8000/recent"

st.set_page_config(page_title="Toxicity Monitor", layout="centered")
st.title("📊 Toxicity Monitor — Lớp học (Realtime)")

refresh_sec = st.sidebar.slider("Refresh interval (s)", min_value=1, max_value=10, value=3)
st.sidebar.markdown("Start audio_capture.py in separate terminal to feed data.")

placeholder = st.empty()

def fetch_stats():
    try:
        r = requests.get(API_STATS, timeout=5)
        if r.ok:
            return r.json()
    except Exception as e:
        return None

while True:
    stats = fetch_stats()
    if stats is None:
        placeholder.error("Không thể kết nối tới API. Chạy uvicorn api:app trước.")
        time.sleep(refresh_sec)
        continue

    total = stats.get("total", 0)
    toxic = stats.get("toxic", 0)
    rate = stats.get("toxic_rate_percent", 0.0)
    recent = stats.get("recent", [])

    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số câu", total)
        col2.metric("Câu độc hại", toxic)
        col3.metric("Tỷ lệ độc hại", f"{rate}%")

        st.markdown("### Các câu gần đây")
        if recent:
            df = pd.DataFrame(recent)
            # show only recent toxic rows with label 'toxic'
            df_display = df[["ts", "speaker", "text", "label", "score"]].head(20)
            st.dataframe(df_display)
        else:
            st.write("Chưa có dữ liệu.")

    time.sleep(refresh_sec)
