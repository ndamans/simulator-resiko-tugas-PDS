import streamlit as st
import numpy as np
import joblib

# Load model dan scaler yang sudah dilatih
model = joblib.load('simulasi_T152/model_risiko_v1.joblib')
scaler = joblib.load('simulasi_T152/scaler_risiko_v1.joblib')

st.title("Simulator Risiko Kegagalan Sistem 🛠️")
st.write("Masukkan parameter mesin untuk mendeteksi tingkat risiko.")

# Input dari pengguna lewat web slider
suhu = st.slider("Suhu Mesin (°C)", min_value=10, max_value=150, value=85)
getaran = st.slider("Getaran Mesin", min_value=1, max_value=20, value=7)

# Sederhana: Cek Drift / Warning jika input di luar jangkauan latihan (Etika MLOps)
if suhu > 120 or suhu < 10:
    st.warning("⚠️ Input di luar jangkauan data latihan yang aman. Hasil mungkin kurang akurat!")

if st.button("Simulasi Risiko"):
    # Preprocessing & Prediksi
    data = np.array([[suhu, getaran]])
    data_scaled = scaler.transform(data)
    prediksi = model.predict(data_scaled)[0]

    st.success(f"Skor Risiko Terdeteksi: {prediksi:.2f}%")
