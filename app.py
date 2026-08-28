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

# Thiết lập chứng chỉ SSL cho mô hình Hugging Face nếu cần
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Big Data Streaming Dashboard", layout="wide")

# --- 1. Khởi tạo mô hình AI ---
@st.cache_resource
def load_sentiment_model():
    """Tải mô hình RoBERTa phân tích cảm xúc, được cache để load 1 lần duy nhất"""
    with st.spinner("Đang tải mô hình Trí tuệ nhân tạo (RoBERTa)..."):
        # Sử dụng mô hình chính xác như bên Colab
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_analyzer = load_sentiment_model()

# --- 2. Cấu hình dữ liệu giả lập (Vì không kết nối Kafka trực tiếp từ xa được) ---
# Danh sách các đánh giá mẫu từ dataset Amazon Fashion
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

# --- 3. Khởi tạo trạng thái lưu trữ (Session State) ---
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0
if "started_at" not in st.session_state:
    st.session_state.started_at = time.monotonic()
if "run_duration" not in st.session_state:
    # Giả lập thời gian chạy tối đa là 1800s như bên Colab
    st.session_state.run_duration = 1800

# --- 4. Hàm tạo biểu đồ 3D Phân phối cảm xúc (base64) ---
def generate_3d_chart_base64(rows):
    """Vẽ biểu đồ cột 3D bằng Matplotlib và mã hóa thành base64 để hiển thị trên Streamlit"""
    categories = ['Rất tích cực', 'Tích cực', 'Trung lập', 'Tiêu cực', 'Rất tiêu cực']
    counts = {cat: 0 for cat in categories}
    
    # Đếm số lượng cảm xúc từ dữ liệu lịch sử
    if rows:
        for row in rows:
            emo = row['emotion']
            if emo in counts:
                counts[emo] += 1
    
    y_vals = [counts[cat] for cat in categories]
    
    fig = plt.figure(figsize=(6.4, 4.6), facecolor='white')
    ax = fig.add_subplot(projection='3d')
    x = np.arange(len(categories))
    y = np.zeros(len(categories))
    z = np.zeros(len(categories))
    dx = np.ones(len(categories)) * 0.4
    dy = np.ones(len(categories)) * 0.4
    dz = y_vals
    
    bar_x = x - 0.2
    # Bảng màu chính xác giống bên Colab
    bar_colors = ['#2b8a3e', '#51cf66', '#ced4da', '#ff6b6b', '#c92a2a']
    
    ax.bar3d(bar_x, y, z, dx, dy, dz, color=bar_colors, shade=True, edgecolor='none', alpha=0.92)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, rotation=30, color='#212529', ha='right')
    ax.set_zlabel('Số lượng', fontsize=10, fontweight='bold', color='#212529')
    ax.set_title('Biểu đồ 3D Phân phối cảm xúc', fontsize=12, fontweight='bold', color='#1d3557', pad=15)
    
    # Định dạng màu nền trục
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.28)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# --- 5. Xây dựng giao diện Streamlit ---

# Tiêu đề ứng dụng
st.markdown("""
    <style>
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
        background: linear-gradient(135deg, #0d6efd, #0a58ca);
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-banner {
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 20px;
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }
    .report-title {
        font-size: 1.2em;
        font-weight: 700;
        color: #212529;
        margin-bottom: 15px;
        text-transform: uppercase;
        border-bottom: 2px solid #0d6efd;
        padding-bottom: 8px;
    }
    .badge {
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h2 class="main-title">Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng</h2>', unsafe_allow_html=True)
st.markdown('<div class="status-banner">TRẠNG THÁI: HỆ THỐNG STREAMING TỐC ĐỘ CAO ĐANG HOẠT ĐỘNG</div>', unsafe_allow_html=True)

# Hiển thị các thẻ chỉ số thống kê
elapsed = time.monotonic() - st.session_state.started_at
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Đã xử lý", value=f"{st.session_state.processed_count:,}")
with col2:
    st.metric(label="Đã chuyển OCI", value=f"{st.session_state.processed_count:,}")
with col3:
    st.metric(label="Trạng thái", value="🟢 Active")
with col4:
    st.metric(label="Thời gian chạy", value=f"{int(min(elapsed, st.session_state.run_duration))}s / {st.session_state.run_duration}s")

st.markdown("---")

# Hiển thị Bảng và Biểu đồ trong 2 cột
left_col, right_col = st.columns([1.5, 1.0])

with left_col:
    st.markdown('<div class="report-title">Bảng đánh giá thời trang gần nhất</div>', unsafe_allow_html=True)
    if st.session_state.history:
        # Chuyển đổi lịch sử thành DataFrame để hiển thị
        df_display = pd.DataFrame(st.session_state.history)
        
        # Hàm định dạng từng dòng trong bảng
        def format_history_row(row):
            stars_str = "⭐" * max(1, min(5, int(row['amazon_rating'])))
            rating_col = f"{row['amazon_rating']} / 5.0<br><span style='font-size: 0.85em; color: #f39c12; letter-spacing: 1px;'>{stars_str}</span>"
            review_col = f"<div style='font-size: 1.1em; white-space: pre-wrap;'>{row['title']}</div>"
            emotion_col = f"<span class='badge' style='background-color: {row['b_color']}; color: {row['t_color']};'>{row['emotion']}</span>"
            return [rating_col, review_col, emotion_col]

        # Áp dụng định dạng và loại bỏ các cột phụ
        df_formatted = df_display[['amazon_rating', 'title', 'emotion', 't_color', 'b_color']].tail(7)
        df_formatted = pd.DataFrame([format_history_row(r) for r in df_formatted.to_dict('records')], columns=['Rating', 'Nội dung phản hồi', 'Phân tích cảm xúc (AI)'])
        
        # Hiển thị bảng HTML
        st.write(df_formatted.to_html(escape=False, index=False, classes='data-table'), unsafe_allow_html=True)
    else:
        st.info("Đang chờ luồng dữ liệu streaming...")

with right_col:
    st.markdown('<div class="report-title">Biểu đồ 3D Phân phối cảm xúc</div>', unsafe_allow_html=True)
    if st.session_state.history:
        chart_uri = generate_3d_chart_base64(st.session_state.history)
        st.image(chart_uri, use_column_width=True)
    else:
        st.write("Chưa đủ dữ liệu để vẽ biểu đồ.")

# --- 6. Vòng lặp cập nhật Real-time bằng `st.rerun()` ---
if elapsed < st.session_state.run_duration:
    # Tự động làm mới trang để tạo hiệu ứng streaming
    time.sleep(1.2) # Giữ nhịp cập nhật
    
    # Tăng bộ đếm và lấy ngẫu nhiên một review mẫu
    st.session_state.processed_count += 1
    rev_text, rating = random.choice(sample_reviews)
    
    # Phân tích cảm xúc thực tế bằng mô hình AI
    res = sentiment_analyzer(rev_text[:500])[0]
    label = res['label'].lower()
    
    # Ánh xạ nhãn AI và rating sang màu sắc và mô tả cảm xúc
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
        
    # Thêm kết quả vào lịch sử
    st.session_state.history.append({
        "amazon_rating": rating,
        "title": rev_text,
        "emotion": emo,
        "t_color": t_color,
        "b_color": b_color
    })
    
    # Kích hoạt làm mới trang
    st.rerun()
else:
    st.success("Quá trình streaming giả lập đã hoàn tất.")