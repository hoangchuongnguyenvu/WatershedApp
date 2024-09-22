import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# CSS cho tiêu đề và hiệu ứng nền động
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    h1 {
        font-size: 48px;
        text-align: center;
        color: #fff;
        opacity: 0;
        animation: fadeIn 3s ease-in forwards;
    }

    @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(-20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Hiệu ứng cho tiêu đề "Kết quả" */
    .result-title {
        font-size: 36px;
        text-align: center;
        color: #fff;
        opacity: 0;
        animation: resultFadeIn 2s ease-in-out forwards;
        transform: scale(0.8);
    }

    @keyframes resultFadeIn {
        0% {
            opacity: 0;
            transform: scale(0.8);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
    <h1>Ứng dụng thuật toán Watershed</h1>
""", unsafe_allow_html=True)

# Tải nhiều ảnh từ máy tính
uploaded_files = st.file_uploader("Chọn nhiều ảnh", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Kiểm tra nếu có file được tải lên
if uploaded_files is not None and len(uploaded_files) > 0:
    # Tạo nút "Áp dụng thuật toán Watershed" với hiệu ứng shake
    if st.button('Áp dụng thuật toán Watershed cho tất cả ảnh'):
        for idx, uploaded_file in enumerate(uploaded_files):
            # Mở ảnh bằng PIL
            image = Image.open(uploaded_file)
            st.write(f"Ảnh {idx + 1}: {uploaded_file.name}")
            st.image(image, caption=f'Ảnh {idx + 1} đã tải lên.', use_column_width=True)
            
            # Chuyển ảnh sang định dạng numpy array
            image_np = np.array(image)
            
            # Tiền xử lý và áp dụng Watershed
            blur = cv2.medianBlur(src=image_np, ksize=3)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            ret, image_thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(image_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)

            dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)
            ret, sure_foreground = cv2.threshold(src=dist_transform, thresh=0.3 * np.max(dist_transform), maxval=255, type=0)

            sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=1)
            sure_foreground = np.uint8(sure_foreground)
            unknown = cv2.subtract(sure_background, sure_foreground)

            ret, marker = cv2.connectedComponents(sure_foreground)
            marker = marker + 1
            marker[unknown == 255] = 0
            watershed_image = cv2.watershed(image=image_np, markers=marker)

            contour, hierarchy = cv2.findContours(image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
            image_vis = image_np.copy()

            for i in range(len(contour)):
                if hierarchy[0][i][3] == -1:
                    cv2.drawContours(image=image_vis, contours=contour, contourIdx=i, color=(255, 0, 0), thickness=2)

            # Hiển thị tiêu đề "Kết quả" trước khi hiển thị kết quả
            st.markdown('<h2 class="result-title">Kết quả</h2>', unsafe_allow_html=True)

            # Hiển thị kết quả sau khi áp dụng thuật toán Watershed
            st.image(image_vis, caption=f'Kết quả ảnh {idx + 1} sau khi áp dụng Watershed.', use_column_width=True)
            st.write(f'Thuật toán Watershed cho ảnh {idx + 1} đã hoàn tất.')

            # Chuyển kết quả sang định dạng ảnh và tải về
            result_img = Image.fromarray(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))

            # Tạo buffer để lưu ảnh tạm thời
            buf = io.BytesIO()
            result_img.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            # Thêm nút tải ảnh về
            st.download_button(
                label=f"Tải ảnh {idx + 1} kết quả",
                data=byte_im,
                file_name=f"result_watershed_{idx + 1}.jpg",
                mime="image/jpeg"
            )
