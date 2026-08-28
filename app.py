import streamlit as st
import time
import random
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Big Data Streaming Dashboard", layout="wide")

st.title("Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng")
st.markdown("Trạng thái: **Hệ thống Streaming thời gian thực đang hoạt động**")

# Tải mô hình AI phân tích cảm xúc (cache để không load lại nhiều lần)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

analyzer = load_sentiment_model()

# Khởi tạo session state lưu trữ lịch sử dữ liệu
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0

sample_reviews = [
    ("Just what I needed. Great pants.", 5.0),
    ("Great price for a great item", 5.0),
    ("Two stars because these run small in the waist and are uncomfortable.", 2.0),
    ("Love these pants, good work pants or to wear casually.", 4.0),
    ("Comfortable and look nice", 5.0)
]

# Các thẻ chỉ số thống kê (Metrics)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Đã xử lý", value=f"{st.session_state.processed_count:,}")
with col2:
    st.metric(label="Đã chuyển OCI", value=f"{st.session_state.processed_count:,}")
with col3:
    st.metric(label="Trạng thái", value="🟢 Active")
with col4:
    st.metric(label="Thời gian chạy", value=f"{st.session_state.processed_count * 1}s")

st.markdown("---")

# Bố cục chia 2 cột: Bảng dữ liệu và Biểu đồ cảm xúc
left_col, right_col = st.columns([1.4, 1.0])

with left_col:
    st.subheader("Bảng đánh giá thời trang gần nhất")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history[-7:])
        st.dataframe(df[['rating', 'review', 'emotion']], use_container_width=True)
    else:
        st.info("Đang chờ dữ liệu streaming...")

with right_col:
    st.subheader("Phân phối cảm xúc")
    if st.session_state.history:
        emotions = [item['emotion'] for item in st.session_state.history]
        emo_counts = {
            "Rất tích cực": emotions.count("Rất tích cực"),
            "Tích cực": emotions.count("Tích cực"),
            "Trung lập": emotions.count("Trung lập"),
            "Tiêu cực": emotions.count("Tiêu cực"),
            "Rất tiêu cực": emotions.count("Rất tiêu cực")
        }
        st.bar_chart(emo_counts)
    else:
        st.write("Chưa có dữ liệu vẽ biểu đồ.")

# Vòng lặp cập nhật real-time tự động làm mới trang
time.sleep(1.5)
st.session_state.processed_count += 1

rev_text, rating = random.choice(sample_reviews)
res = analyzer(rev_text[:500])[0]
label = res['label'].lower()

if "pos" in label:
    emo = "Rất tích cực" if rating >= 4.5 else "Tích cực"
elif "neg" in label:
    emo = "Rất tiêu cực" if rating <= 1.5 else "Tiêu cực"
else:
    emo = "Trung lập"
    
st.session_state.history.append({
    "rating": f"{rating} / 5.0",
    "review": rev_text,
    "emotion": emo
})

st.rerun()