import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import gdown

# 🔽 Tải mô hình từ Google Drive nếu chưa có
file_id = '1nBdEoBfxGHgRyITgFlLRDZn9SdXrBTgS'
model_url = f'https://drive.google.com/uc?id={file_id}'
model_file = 'animal_classifier_advanced.h5'

if not os.path.exists(model_file):
    with st.spinner('Đang tải mô hình...'):
        gdown.download(model_url, model_file, quiet=False)

# 🔍 Tải mô hình
model = tf.keras.models.load_model(model_file)
class_names = ['bird', 'cat', 'dog']

# 📸 Hàm xử lý ảnh
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array

# 🚀 Giao diện Streamlit
st.set_page_config(page_title="Phân loại Động vật", layout="centered")
st.title("📷 Phân loại Ảnh: Chó 🐶, Mèo 🐱, Chim 🐦")
st.write("Tải lên một ảnh để hệ thống phân loại:")

uploaded_file = st.file_uploader("🖼️ Chọn ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image, img_array = preprocess_image(uploaded_file)

    # Hiển thị ảnh
    st.image(image, caption="Ảnh đã chọn", use_column_width=True)

    # Dự đoán
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100

    # Kết quả
    st.subheader(f"✅ Dự đoán: **{class_names[predicted_class].capitalize()}**")
    st.write(f"🎯 Độ chính xác: `{confidence:.2f}%`")
