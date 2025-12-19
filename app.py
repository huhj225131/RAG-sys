# admin_app.py
import streamlit as st
import controller as pipe

st.set_page_config(page_title="RAG Admin Dashboard", layout="wide")

st.title("Hệ thống Quản trị Dữ liệu RAG")
st.markdown("---")

col1, col2 = st.columns(2)

# --- PANEL 1: CẬP NHẬT WEB ---
with col1:
    st.header("🌐 Cập nhật từ Website")
    if st.button("Chạy Auto-Crawl & Embed", type="primary"):
        with st.spinner("Đang kết nối Crawler... Vui lòng đợi (quá trình này có thể mất vài phút)"):
            try:
                success, msg = pipe.update_web_data()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")

with col2:
    st.header("📂 Nạp dữ liệu File")
    st.write("Hỗ trợ định dạng: `.docx`, `.md`")
    
    uploaded_files = st.file_uploader("Upload file", 
                                      type=['docx', 'md'], 
                                      accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"Xử lý {len(uploaded_files)} file"):
            progress_bar = st.progress(0)
            with st.spinner("Đang đọc và vector hóa dữ liệu..."):
                try:
                    # Gọi controller xử lý
                    result_msg = pipe.process_uploaded_files(uploaded_files)
                    progress_bar.progress(100)
                    st.success(result_msg)
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")

st.markdown("---")
st.caption("VNBrain")