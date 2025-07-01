import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (relatif dari posisi file ini di /src/)
model = tf.keras.models.load_model("src/CNN Crash Car Model.h5")

# Class names
class_names = ['01–minor', '02–moderate', '03–severe']

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict():
    st.markdown("<h1 style='text-align: center;'>Deteksi Kerusakan Mobil</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Silakan upload gambar mobil rusak", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        # Tombol prediksi
        if st.button("Predict"):
            with st.spinner("Sedang memproses gambar..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
            
            st.markdown(f"<h2 style='text-align: center; color: blue;'>Prediksi: {predicted_class}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    predict()
