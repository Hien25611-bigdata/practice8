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

# Thiết lập chứng chỉ SSL cho môi trường cloud
os.environ['SSL_CERT_FILE'] = certifi.where()

# Cấu hình giao diện Streamlit toàn trang (Wide mode)
st.set_page_config(page_title="Big Data Streaming Dashboard", layout="wide")

# --- 1. Tải mô hình AI (Cache để không bị load lại nhiều lần) ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_analyzer = load_sentiment_model()

# --- 2. Dữ liệu mẫu đánh giá thời trang ---
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
    ("Poor quality fabric, started pilling after one wash.", 1.0),
    ("Perfect fit and very comfortable material.", 5.0),
    ("Not the color I ordered, disappointed.", 2.0),
    ("Good value for money, stylish design.", 4.0)
]

# --- 3. Khởi tạo trạng thái phiên làm việc (Session State) ---
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0
if "started_at" not in st.session_state:
    st.session_state.started_at = time.monotonic()

run_duration = 1800 # Giả lập thời gian chạy 1800 giây

# --- 4. Hàm vẽ biểu đồ cột 3D phân phối cảm xúc ---
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
    
    bar_x = x - 0.2
    # Màu sắc cột biểu đồ đồng bộ chuẩn với Colab
    bar_colors = ['#2b8a3e', '#51cf66', '#ced4da', '#ff6b6b', '#c92a2a']
    
    ax.bar3d(bar_x, y, z, dx, dy, dz, color=bar_colors, shade=True, edgecolor='none', alpha=0.92)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=30, color='#212529', ha='right')
    ax.set_zlabel('Số lượng', fontsize=9, fontweight='bold', color='#212529')
    ax.set_title('Biểu đồ 3D Phân phối cảm xúc', fontsize=11, fontweight='bold', color='#1d3557', pad=12)
    
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.25)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# --- 5. Giao diện trang web Dashboard ---
st.markdown("""
    <style>
    .main-title {
        color: #ffffff;
        background: linear-gradient(135deg, #0d6efd, #0a58ca);
        padding: 22px;
        border-radius: 12px;
        text-transform: uppercase;
        text-align: center;
        font-weight: 700;
        font-size: 1.4em;
        margin-bottom: 15px;
    }
    .status-banner {
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: 600;
        background-color: #d1e7dd;
        color: #0f5132;
        margin-bottom: 20px;
        border: 1px solid #badbcc;
    }
    .badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.95em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng</div>', unsafe_allow_html=True)
st.markdown('<div class="status-banner">TRẠNG THÁI: HỆ THỐNG STREAMING THỜI GIAN THỰC ĐANG HOẠT ĐỘNG</div>', unsafe_allow_html=True)

# Các thẻ Metrics thống kê
elapsed = time.monotonic() - st.session_state.started_at
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Đã xử lý", value=f"{st.session_state.processed_count:,}")
with col2:
    st.metric(label="Đã chuyển OCI", value=f"{st.session_state.processed_count:,}")
with col3:
    st.metric(label="Trạng thái", value="🟢 Active")
with col4:
    st.metric(label="Thời gian chạy", value=f"{int(min(elapsed, run_duration))}s / {run_duration}s")

st.markdown("---")

# Bố cục 2 cột (Bảng chi tiết và Biểu đồ 3D)
left_col, right_col = st.columns([1.5, 1.0])

with left_col:
    st.subheader("Bảng đánh giá thời trang gần nhất")
    if st.session_state.history:
        df_display = pd.DataFrame(st.session_state.history[-7:])
        
        # Tạo định dạng HTML cho từng dòng trong bảng
        html_table = """
        <table style="width:100%; border-collapse: collapse; background:#ffffff; font-size:0.95em; color: #212529;">
            <thead>
                <tr style="background-color: #212529; color: #ffffff;">
                    <th style="padding: 10px; text-align: center; width: 25%;">Rating</th>
                    <th style="padding: 10px; text-align: left; width: 40%;">Nội dung phản hồi</th>
                    <th style="padding: 10px; text-align: left; width: 35%;">Phân tích cảm xúc (AI)</th>
                </tr>
            </thead>
            <tbody>
        """
        for r in reversed(df_display.to_dict('records')):
            stars = "⭐" * max(1, min(5, int(r['amazon_rating'])))
            html_table += f"""
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 10px; text-align: center; font-weight: bold;">
                        {r['amazon_rating']} / 5.0<br><span style="color: #f39c12; font-size: 0.85em;">{stars}</span>
                    </td>
                    <td style="padding: 10px;">{r['title']}</td>
                    <td style="padding: 10px;">
                        <span class="badge" style="background-color: {r['b_color']}; color: {r['t_color']};">{r['emotion']}</span>
                    </td>
                </tr>
            """
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("Đang chờ dữ liệu streaming...")

with right_col:
    st.subheader("Biểu đồ 3D Phân phối cảm xúc")
    if st.session_state.history:
        chart_uri = generate_3d_chart_base64(st.session_state.history)
        # Sử dụng đúng cú pháp container width tương thích phiên bản Streamlit mới
        st.image(chart_uri, use_container_width=True)
    else:
        st.write("Chưa đủ dữ liệu vẽ biểu đồ.")

# --- 6. Cơ chế tự động làm mới Real-time liên tục ---
if elapsed < run_duration:
    time.sleep(1.2) # Nhịp làm mới mỗi 1.2 giây
    st.session_state.processed_count += 1
    
    # Lấy dữ liệu mẫu tiếp theo
    rev_text, rating = random.choice(sample_reviews)
    res = sentiment_analyzer(rev_text[:500])[0]
    label = res['label'].lower()
    
    # Gắn màu sắc và phân loại cảm xúc chuẩn xác
    if "pos" in label:
        if rating >= 4.5:
            emo, t_color, b_color = ('Rất tích cực', '#052c11', '#a3cfbb')
        else:
            emo, t_color, b_color = ('Tích cực', '#0f5132', '#d1e7dd')
    elif "neg" in label:
        if rating <= 1.5:
            emo, t_color, b_color = ('Rất tiêu cực', '#58151c', '#f1aeb5')
        else:
            emo, t_color, b_color = ('Tiêu cực', '#842029', '#f8d7da')
    else:
        emo, t_color, b_color = ('Trung lập', '#41464b', '#e2e3e5')
        
    st.session_state.history.append({
        "amazon_rating": rating,
        "title": rev_text,
        "emotion": emo,
        "t_color": t_color,
        "b_color": b_color
    })
    
    # Kích hoạt làm mới trang liên tục giữ hiệu ứng real-time
    st.rerun()