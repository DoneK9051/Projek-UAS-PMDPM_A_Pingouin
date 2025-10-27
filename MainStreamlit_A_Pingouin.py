import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px

# === IMPORT PINTAR (HIBRID) ===
try:
    import tflite_runtime.interpreter as tflite

    print("Berhasil impor 'tflite_runtime' (mode server/deploy)")
except ImportError:
    import tensorflow as tf

    tflite = tf.lite
    print("Gagal impor 'tflite_runtime', menggunakan 'tf.lite' (mode lokal)")
# === SELESAI IMPORT HIBRID ===


# === FUNGSI UNTUK MEMUAT CSS ===
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS '{file_name}' tidak ditemukan.")


# === KONFIGURASI APLIKASI ===
st.set_page_config(
    page_title="🍜 Klasifikasi Makanan Nusantara",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === CSS FILE ===
load_css("style.css")


# === NAMA KELAS DAN EMOJI ===
def get_class_names():
    return ["gudeg", "papeda", "pempek", "rendang"]


def get_food_emoji(pred_class):
    emojis = {"gudeg": "🥘", "papeda": "🥣", "pempek": "🐟", "rendang": "🥩"}
    return emojis.get(pred_class, "❓")


# === FUNGSI MODEL (TFLITE) ===
MODEL_PATH = "BestModel_CostumCNN.tflite"


@st.cache_resource
def load_model():
    """Memuat TFLite interpreter dan mengalokasikan tensor."""
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Gagal memuat model TFLite: {e}")
        st.error(f"Pastikan file '{MODEL_PATH}' ada di folder yang sama.")
        return None, None, None


def preprocess_image(image, target_size=(128, 128)):
    """Pre-processing gambar agar sesuai dengan input model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    return img_array


# === UI APLIKASI ===
# --- HEADER ---
st.markdown(
    "<h1 class='main-header'>🍜 Klasifikasi Makanan Nusantara</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-header'>Unggah gambar makanan Anda 📸 dan biarkan AI kami menebaknya!</p>",
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Tentang Proyek")
    st.markdown(
        """
        Ini adalah prototipe aplikasi web untuk Ujian Tengah Semester (UTS) .
        
        - **Model:** Custom CNN (TFLite)
        - **Tujuan:** Klasifikasi 4 Makanan Nusantara.
        - **Author:** Beny, Denis, Renaldi
        """
    )
    st.divider()
    st.info("Akurasi model mungkin tidak 100%.")

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])
prob = None
interpreter, input_details, output_details = load_model()

with col1:
    st.header("1. Unggah Gambar Anda 🖼️")
    uploaded_file = st.file_uploader(
        "Pilih file gambar...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    with st.expander("ℹ️ Info Kelas yang Dilatih"):
        st.write(
            """
            Model ini dilatih untuk mengenali 4 kelas makanan:
            - 🥘 **gudeg**
            - 🥣 **papeda**
            - 🐟 **pempek**
            - 🥩 **rendang**
            """
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

with col2:
    st.header("2. Hasil Prediksi AI 🧠")

    if uploaded_file is not None:
        if interpreter:
            with st.spinner("AI sedang berpikir... 🤖"):

                input_shape = input_details[0]["shape"]
                target_size = (input_shape[1], input_shape[2])

                img_array = preprocess_image(image, target_size=target_size)

                interpreter.set_tensor(input_details[0]["index"], img_array)
                interpreter.invoke()
                prob = interpreter.get_tensor(output_details[0]["index"])[0]

                pred_index = np.argmax(prob)
                class_names = get_class_names()

                if len(prob) != len(class_names):
                    st.error(
                        f"Error: Model output {len(prob)} kelas, tapi daftar kelas punya {len(class_names)}."
                    )
                    prob = None
                else:
                    pred_class = class_names[pred_index]
                    confidence = np.max(prob) * 100
                    emoji = get_food_emoji(pred_class)

                    st.metric(
                        label="Hasil Tebakan AI:",
                        value=f"{emoji} {pred_class.capitalize()}",
                        delta=f"Keyakinan {confidence:.2f}%",
                        delta_color="normal",
                    )

                    if confidence > 80:
                        st.success("🎯 Prediksi sangat yakin!")
                    elif confidence > 60:
                        st.info("👍 Prediksi cukup yakin")
                    elif confidence > 40:
                        st.warning("⚠️ Prediksi kurang yakin")
                    else:
                        st.error("❌ Prediksi sangat rendah, coba gambar lain")
        else:
            st.error("Model tidak dapat dimuat. Cek log.")
    else:
        st.info("Silakan unggah gambar di sebelah kiri untuk melihat hasil prediksi.")

# === GRAFIK PROBABILITAS ===
if prob is not None:
    st.divider()
    st.subheader("📊 Distribusi Probabilitas")

    class_names = get_class_names()
    prob_data = pd.DataFrame(
        {"Kelas": class_names, "Probabilitas": prob * 100}
    ).sort_values("Probabilitas", ascending=False)

    fig = px.bar(
        prob_data,
        x="Kelas",
        y="Probabilitas",
        color="Probabilitas",
        color_continuous_scale="RdYlGn",
        text_auto=".2f",
        title="Keyakinan Model untuk Setiap Kelas (%)",
    )
    fig.update_layout(yaxis_title="Probabilitas (%)")
    st.plotly_chart(fig, use_container_width=True)
