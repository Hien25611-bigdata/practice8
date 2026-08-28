import streamlit as st
import time

st.title("Big Data Streaming Real-time Dashboard")

# Tạo vùng chứa dữ liệu động
placeholder = st.empty()

for seconds in range(200):
    with placeholder.container():
        st.write(f"Đang cập nhật dữ liệu thời gian thực... Thời gian: {seconds}s")
        # Tại đây bạn đọc dữ liệu mới nhất từ Database hoặc file log
        # st.dataframe(df_latest)
    time.sleep(1)