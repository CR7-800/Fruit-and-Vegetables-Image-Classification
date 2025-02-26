import streamlit as st
import numpy as np
import tensorflow as tf
import time
from skimage import io
from skimage.transform import resize
import json

# 設定 Streamlit 頁面樣式
st.set_page_config(page_title="🍎 蔬果圖片辨識 🍌", layout="centered")

# 自訂 CSS 讓 UI 更美觀
st.markdown(
    """
    <style>
    .stTitle {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #FFD700; /* 金色 */
        margin-bottom: 20px;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #222;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #00FF7F; /* 螢光綠 */
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 頁面標題
st.markdown("<div class='stTitle'>🍎 蔬果圖片辨識 🍌</div>", unsafe_allow_html=True)

# 上傳圖片
uploaded_file = st.file_uploader(
    "上傳圖片 (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"], help="拖放或點擊選擇圖片"
)

img_size = (224, 224, 3)

if uploaded_file is not None:
    # 顯示上傳的圖片
    st.image(uploaded_file, caption="已上傳的圖片", use_container_width=True)

    # 讀取圖片並處理
    image = io.imread(uploaded_file)
    image = resize(image, img_size[:-1])    
    X1 = image.reshape(1, *img_size)

    # **使用載入動畫與進度條**
    with st.spinner('🔍 模型正在辨識圖片中，請稍候...'):
        progress_bar = st.progress(0)

        # 模擬處理時間，分段更新進度條
        for percent_complete in range(0, 101, 10):
            time.sleep(0.1)  # 模擬計算延遲
            progress_bar.progress(percent_complete)

        # 載入模型
        model = tf.keras.models.load_model("fruit_model.keras", compile=False)

        # 預測結果
        predictions = model.predict(X1)
        confidence = np.max(predictions) * 100  # 轉換為百分比
        keys = list(json.load(open("label_dict.json", "r", encoding="utf-8")).keys())
        label = "未知" if np.argmax(predictions) >= len(keys) else keys[np.argmax(predictions)].split("_")[1]

    # **清除進度條**
    progress_bar.empty()

    # 顯示結果
    st.markdown(f"""
    <div class='result-box'>
        預測結果: {label} <br> 🔍 信心分數: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)


