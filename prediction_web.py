# streamlit run d:/03_fruit/prediction_web.py

import streamlit as st 
import numpy as np  
import tensorflow as tf
from skimage import io
from skimage.transform import resize
import json

# 設定 Streamlit 頁面樣式
st.set_page_config(page_title="蔬果圖片辨識", layout="centered")

# 自訂 CSS 讓頁面更美觀
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #ff7f50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .stTitle {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #ffcc00;
    }
    .result-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #222;
        text-align: center;
    }
    .stFileUploader {
        background-color: #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 載入標籤對應表
json_path = r"D:\03_fruit\label_dict.json"

with open(json_path, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)

# 載入模型
model = tf.keras.models.load_model('D:\\03_fruit\\fruit_model.keras', compile=False)

# 頁面標題
st.markdown("<div class='stTitle'>🍎 蔬果圖片辨識 🍌</div>", unsafe_allow_html=True)

# 上傳圖片
uploaded_file = st.file_uploader("上傳圖片 (.jpg, .png)", type=["jpg", "png"], help="Drag and drop or click to select an image", label_visibility='visible')
img_size = (224, 224, 3)

if uploaded_file is not None:
    # 顯示上傳的圖片
    st.image(uploaded_file, caption="已上傳的圖片", use_container_width=True)
    
    # 讀取圖片並處理
    image = io.imread(uploaded_file)
    image = resize(image, img_size[:-1])    
    X1 = image.reshape(1, *img_size)
    
    # 預測結果
    predictions = np.argmax(model.predict(X1))
    
    # 獲取標籤
    keys = list(label_dict.keys())
    label = "未知" if predictions >= len(keys) else keys[predictions].split("_")[1]
    
    # 顯示結果
    st.markdown(f"""
    <div class='result-box'>
        <h2>預測結果: {label}</h2>
    </div>
    """, unsafe_allow_html=True)


