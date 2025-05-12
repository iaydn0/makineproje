
import streamlit as st
import joblib
import numpy as np

# Model ve scaler'ları yükle
model = joblib.load("model.pkl")
scaler_amount, scaler_time = joblib.load("scaler.pkl")

st.title("💳 Kredi Kartı Dolandırıcılığı Tahmini")

st.markdown("Aşağıdaki bilgileri girerek işlemin dolandırıcılık olup olmadığını tahmin edin.")

amount = st.number_input("İşlem Tutarı (Amount)", min_value=0.0)
time = st.number_input("İşlemin gerçekleştiği zaman (saniye cinsinden)", min_value=0.0)

input_data = np.zeros((1, 30))
input_data[0, -2] = scaler_time.transform([[time]])[0][0]
input_data[0, -1] = scaler_amount.transform([[amount]])[0][0]

if st.button("Tahmin Et"):
    prediction = model.predict(input_data)[0]
    st.success("✅ İşlem muhtemelen NORMAL.") if prediction == 0 else st.error("⚠️ İşlem DOLANDIRICILIK olabilir!")
