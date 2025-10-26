import tensorflow as tf
import os

MODEL_H5_PATH = "BestModel_CostumCNN_Pingouin.h5"
MODEL_TFLITE_PATH = "BestModel_CostumCNN.tflite"

print(f"Memuat model dari: {MODEL_H5_PATH}")
model = tf.keras.models.load_model(MODEL_H5_PATH)
print("Model .h5 berhasil dimuat.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("Mengonversi model ke TFLite (TANPA optimasi, akurasi penuh)...")
tflite_model = converter.convert()

print(f"Menyimpan model TFLite baru ke: {MODEL_TFLITE_PATH}")
with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("Konversi Selesai! Model .tflite baru (presisi penuh) telah disimpan.")
