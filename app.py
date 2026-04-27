import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

st.title("🔧 Deteksi Defect Casting")

@st.cache_resource
def load_model():
    model = tf.saved_model.load('saved_model_hf_v2')
    infer = model.signatures['serving_default']
    return infer

infer = load_model()

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Pilih gambar produk casting", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Anda', width=400)

    if st.button("🔍 Prediksi"):
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        output = infer(tf.constant(img_array))
        pred = output[next(iter(output))].numpy()[0][0]

        label = idx_to_class[1] if pred > 0.5 else idx_to_class[0]
        confidence = pred if pred > 0.5 else 1 - pred

        st.success(f"**HASIL: {label.upper()}**")
        st.info(f"Confidence: {confidence:.2%}")
