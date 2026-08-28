import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Big Data Streaming Dashboard", layout="wide")

st.title("Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng")

# Tải mô hình AI phân tích cảm xúc
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_analyzer = load_model()

# Tạo giao diện nhập liệu hoặc hiển thị luồng dữ liệu
text_input = st.text_area("Nhập nội dung phản hồi của khách hàng:", "Sản phẩm dùng rất tốt, tôi rất thích!")
if st.button("Phân tích ngay"):
    result = sentiment_analyzer(text_input[:500])[0]
    st.write(f"Kết quả phân tích: **{result['label']}** (Độ tin cậy: {result['score']:.2f})")