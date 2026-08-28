import streamlit as st
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import io
import base64
import certifi
import os

os.environ['SSL_CERT_FILE'] = certifi.where()

st.set_page_config(page_title="Big Data Streaming Dashboard", layout="wide")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_analyzer = load_sentiment_model()

sample_reviews = [
    ("Just what I needed. Great pants.", 5.0),
    ("Great price for a great item", 5.0),
    ("Two stars because these run small in the waist and are uncomfortable.", 2.0),
    ("Love these pants, good work pants or to wear casually.", 4.0),
    ("Comfortable and look nice", 5.0),
    ("Too big. Returning.", 2.0),
    ("Excellent quality and fast delivery.", 5.0),
    ("A bit tight, but okay.", 3.0),
    ("Exactly what I expected, very happy with the purchase.", 5.0),
    ("Poor quality fabric, started pilling after one wash.", 1.0)
]

if "history" not in st.session_state:
    st.session_state.history = []
if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0
if "started_at" not in st.session_state:
    st.session_state.started_at = time.monotonic()

run_duration = 1800

def generate_3d_chart_base64(rows):
    categories = ['Rất tích cực', 'Tích cực', 'Trung lập', 'Tiêu cực', 'Rất tiêu cực']
    counts = {cat: 0 for cat in categories}
    if rows:
        for row in rows:
            emo = row['emotion']
            if emo in counts:
                counts[emo] += 1
    y_vals = [counts[cat] for cat in categories]
    
    fig = plt.figure(figsize=(6.2, 4.2), facecolor='white')
    ax = fig.add_subplot(projection='3d')
    x = np.arange(len(categories))
    y = np.zeros(len(categories))
    z = np.zeros(len(categories))
    dx = np.ones(len(categories)) * 0.4
    dy = np.ones(len(categories)) * 0.4
    dz = y_vals
    
    bar_colors = ['#2b8a3e', '#51cf66', '#ced4da', '#ff6b6b', '#c92a2a']
    ax.bar3d(x - 0.2, y, z, dx, dy, dz, color=bar_colors, shade=True, edgecolor='none', alpha=0.92)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=30, color='#212529', ha='right')
    ax.set_zlabel('Số lượng', fontsize=9, fontweight='bold')
    ax.set_title('Biểu đồ 3D Phân phối cảm xúc', fontsize=11, fontweight='bold', color='#1d3557', pad=12)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

st.title("Ứng dụng Big Data Streaming phân tích độ hài lòng")
st.info("Trạng thái: Hệ thống Streaming thời gian thực đang hoạt động")

elapsed = time.monotonic() - st.session_state.started_at
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Đã xử lý", f"{st.session_state.processed_count:,}")
with col2:
    st.metric("Đã chuyển OCI", f"{st.session_state.processed_count:,}")
with col3:
    st.metric("Trạng thái", "🟢 Active")
with col4:
    st.metric("Thời gian", f"{int(min(elapsed, run_duration))}s / {run_duration}s")

st.markdown("---")

left_col, right_col = st.columns([1.5, 1.0])

with left_col:
    st.subheader("Bảng đánh giá gần nhất")
    if st.session_state.history:
        df_display = pd.DataFrame(st.session_state.history[-7:][::-1])
        # Hiển thị bằng bảng trực quan của Streamlit
        st.dataframe(
            df_display[['amazon_rating', 'title', 'emotion']],
            column_config={
                "amazon_rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
                "title": "Nội dung phản hồi",
                "emotion": "Phân tích cảm xúc (AI)"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.write("Đang chờ dữ liệu...")

with right_col:
    st.subheader("Biểu đồ 3D Cảm xúc")
    if st.session_state.history:
        chart_uri = generate_3d_chart_base64(st.session_state.history)
        st.image(chart_uri, use_container_width=True)
    else:
        st.write("Chưa có dữ liệu biểu đồ.")

if elapsed < run_duration:
    time.sleep(1.2)
    st.session_state.processed_count += 1
    rev_text, rating = random.choice(sample_reviews)
    res = sentiment_analyzer(rev_text[:500])[0]
    label = res['label'].lower()
    
    if "pos" in label:
        emo = "Rất tích cực" if rating >= 4.5 else "Tích cực"
    elif "neg" in label:
        emo = "Rất tiêu cực" if rating <= 1.5 else "Tiêu cực"
    else:
        emo = "Trung lập"
        
    st.session_state.history.append({
        "amazon_rating": rating,
        "title": rev_text,
        "emotion": emo
    })
    st.rerun()