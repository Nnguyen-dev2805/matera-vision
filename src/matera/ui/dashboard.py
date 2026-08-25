import os
import tempfile
from pathlib import Path

import streamlit as st

from matera.data.extract import extract_pages
from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.vision.alignment import align_page
from matera.vision.mark import extract_mark_scores
from matera.vision.routing import route_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.ui.components import draw_debug_boxes, result_to_dataframe

# Cấu hình trang
st.set_page_config(page_title="Matera Vision Debug", layout="wide")

st.title("Matera Vision - Pipeline Debugger")

# Sidebar cho các công cụ điều khiển
st.sidebar.header("Tải lên PDF")
uploaded_file = st.sidebar.file_uploader("Chọn file PDF", type=["pdf"])

if uploaded_file is not None:
    # Lưu file tạm để xử lý
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = Path(tmp_file.name)
        
    st.sidebar.success("Đã tải file thành công!")
    st.sidebar.write(f"Đang xử lý: `{uploaded_file.name}`")
    
    # [Task 2] Extract Pages
    with st.spinner("Đang trích xuất các trang..."):
        try:
            pages = list(extract_pages(pdf_path))
        except Exception as e:
            st.error(f"Lỗi trích xuất PDF: {e}")
            st.stop()
            
    if not pages:
        st.error("Không tìm thấy trang nào trong PDF.")
        st.stop()
        
    page_count = len(pages)
    
    page_options = [i + 1 for i in range(page_count)]
    
    # Page selector
    selected_page_nums = st.sidebar.multiselect(
        "Chọn các trang để Debug", 
        options=page_options,
        default=[1] if page_count >= 1 else []
    )
    
    if not selected_page_nums:
        st.info("Vui lòng chọn ít nhất một trang để xem kết quả.")
        st.stop()
        
    for selected_page_num in selected_page_nums:
        st.markdown(f"## Kết quả Debug - Trang {selected_page_num}")
        
        # Process the selected page
        selected_page = pages[selected_page_num - 1]
    
        with st.spinner(f"Đang phân tích trang {selected_page_num}..."):
            try:
                # 1. Load profiles & reference image
                from PIL import Image
                
                layout_profile = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
                semantic_profile = load_semantic_profile(Path("profiles/matera-pre/v1/semantic.json"))
                classifier_path = Path("models/matera-pre-v1/classifier.joblib")
                
                ref_image_path = "scratch/synthetic_median_reference.png"
                reference_image = Image.open(ref_image_path)
                
                # 2. Align Page
                align_config = AlignmentConfig(
                    algorithm="orb",
                    transform_model="affine",
                    inlier_threshold=0.05
                )
                aligned_page = align_page(selected_page, reference_image, align_config)
                
                # 3. Extract Scores
                debug_dir = Path("data/debug/evidence")
                debug_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np
                import cv2
                
                orig_arr = np.array(aligned_page.image)
                debug_full_mask = np.zeros_like(orig_arr)
                scores = extract_mark_scores(
                    aligned_page, 
                    semantic_profile, 
                    layout_profile.pages[0], 
                    reference_image, 
                    debug_dir=str(debug_dir),
                    debug_full_mask=debug_full_mask
                )
                
                # 4. Route (Baseline deterministic only)
                route_config = RoutingConfig()
                result = route_page(scores, semantic_profile, selected_page_num, route_config)
                
                # [Task 3] Vẽ Debug Box
                draw_debug_boxes(aligned_page, result, layout_profile.pages[0])
                
                # Layout hiển thị: Trái (Ảnh Debug), Phải (Bảng Kết quả)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("1. Bản Gốc (Đã Nắn & Chấm)")
                    st.image(aligned_page.image, use_container_width=True)
                    
                with col2:
                    st.subheader("2. X-Quang Mặt Nạ (Debug Layer 1)")
                    st.image(cv2.cvtColor(debug_full_mask, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                st.subheader("Bảng Kết Quả Trạng Thái")
                df = result_to_dataframe(result, semantic_profile, scores)
                st.dataframe(df, use_container_width=True)
                    
                st.divider() # Ngăn cách giữa các trang
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình phân tích trang {selected_page_num}: {e}")
                import traceback
                st.code(traceback.format_exc())
            
else:
    st.info("Vui lòng tải lên một file PDF ở thanh bên trái để bắt đầu.")
