# -*- coding: utf-8 -*-
import os
import tempfile
import cv2
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# 🚨 必须第一个 Streamlit 命令
st.set_page_config(page_title="红外无人机检测系统", layout="wide")

# ====================== 基础配置 ======================
MODEL_PATH = r"C:\Users\13579\Desktop\cv\weights\best_wyc.pt"

# 加载模型
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ====================== 页面标题和说明 ======================
st.title("🚁 YOLOv8 红外无人机数量检测系统")

st.markdown("""
本系统基于 **YOLOv8 + Streamlit** 实现，支持无人机目标检测的 **图片检测** 与 **视频检测** 两种模式。  
您可以通过左侧参数栏调整检测置信度，上传图片或视频后点击 **开始检测** 按钮，即可查看检测效果。  

📌 **使用指南：**  
1. 在左侧 **参数配置栏** 调整检测置信度（数值越高，检测越严格）。  
2. 在左侧选择检测模式：**图片检测** 或 **视频检测**。  
3. 点击中间区域上传文件（支持常见格式：JPG、PNG、MP4 等）。  
4. 点击 **开始检测** 按钮，结果将在界面右侧展示，并可下载保存。  

⚠️ 注意：上传文件大小需小于 **200MB**。
""")

st.sidebar.header("⚙️ 参数配置")
confidence = st.sidebar.slider("检测置信度", 0.25, 1.0, 0.5, 0.01)

mode = st.sidebar.radio("选择模式", ["图片检测", "视频检测"])

# ====================== 图片检测 ======================
if mode == "图片检测":
    uploaded_img = st.file_uploader("📷 上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_img is not None:
        image = PIL.Image.open(uploaded_img)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ 原始图片")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 检测结果")
            if "detected_img" in st.session_state:
                st.image(st.session_state["detected_img"], use_container_width=True)
                st.success(f"✅ 检测完成，共检测到目标数量: **{st.session_state['detected_count']}** 架无人机")

                # 下载按钮
                with open("detected_result.jpg", "wb") as f:
                    f.write(cv2.imencode(".jpg", st.session_state["detected_img"])[1].tobytes())
                with open("detected_result.jpg", "rb") as f:
                    st.download_button("💾 下载检测结果图片", f, file_name="detected_result.jpg")

            else:
                st.info("请点击下方按钮开始检测")

        # 检测按钮放到下方居中
        btn_col = st.columns([1, 2, 1])
        with btn_col[1]:
            if st.button("🚀 开始检测", use_container_width=True):
                results = model.predict(image, conf=confidence)
                res_plot = results[0].plot()
                st.session_state["detected_img"] = res_plot
                st.session_state["detected_count"] = len(results[0].boxes)
                st.rerun()

# ====================== 视频检测 ======================
elif mode == "视频检测":
    uploaded_vid = st.file_uploader("🎥 上传视频", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_vid is not None:
        # 保存临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())

        st.markdown("### 🔎 检测过程")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📹 原始视频帧")
            raw_frame_placeholder = st.empty()   # 原始帧占位符

        with col2:
            st.subheader("🎯 检测结果帧")
            det_frame_placeholder = st.empty()   # 检测帧占位符
            info_box = st.empty()

        # 检测按钮放到下方居中
        btn_col = st.columns([1, 2, 1])
        with btn_col[1]:
            if st.button("🚀 开始检测", use_container_width=True):
                cap = cv2.VideoCapture(tfile.name)
                frame_count = 0
                total_objects = 0

                # 保存检测结果视频
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = "detected_video.mp4"
                out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    # 推理
                    results = model.predict(frame, conf=confidence, verbose=False)
                    res_plotted = results[0].plot()

                    # 当前帧目标数量
                    current_count = len(results[0].boxes)
                    total_objects += current_count

                    # 左边显示原始帧
                    raw_frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    # 右边显示检测结果
                    det_frame_placeholder.image(res_plotted, channels="BGR", use_container_width=True)

                    # 帧信息
                    info_box.info(f"帧 {frame_count} - 检测到目标数量: {current_count}")

                    # 保存检测帧到视频
                    out.write(res_plotted)

                cap.release()
                out.release()

                st.success(f"✅ 检测完成，总帧数: {frame_count}, 累计检测目标数量: {total_objects}")

                # 提供下载按钮
                with open(out_path, "rb") as f:
                    st.download_button("💾 下载检测结果视频", f, file_name="detected_video.mp4")
