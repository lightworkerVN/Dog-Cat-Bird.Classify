import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🖼️ Phân loại ảnh với mô hình của bạn")

MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1nBdEoBfxGHgRyITgFlLRDZn9SdXrBTgS"

# Tải model nếu chưa có
if not os.path.exists(MODEL_PATH):
    with st.spinner("Đang tải model từ Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
st.success("✅ Model đã sẵn sàng!")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên để phân loại", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Xử lý ảnh cho đúng input của model (giả sử là 224x224)
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Dự đoán
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.write(f"🔍 Dự đoán: **Lớp {predicted_class}**")
