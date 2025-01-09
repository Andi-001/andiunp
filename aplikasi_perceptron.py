import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Memuat model dan scaler
perceptron_model = pickle.load(open('perceptron_model.pkl', 'rb'))
scaler = pickle.load(open('sc.pkl', 'rb'))

# Fungsi untuk melakukan prediksi dan menampilkan epoch, bias, dan weight
def predict(input_data, epoch=10):
    # Normalisasi input data
    input_data_scaled = scaler.transform([input_data])

    # Melakukan prediksi dengan model Perceptron
    prediction = perceptron_model.predict(input_data_scaled)

    # Menampilkan informasi epoch, bias, dan weight
    weights = perceptron_model.coef_
    bias = perceptron_model.intercept_

    return prediction, epoch, bias, weights

# Judul aplikasi
st.title("Prediksi dengan Model Perceptron")

# Input data oleh pengguna
st.write("Masukkan data untuk prediksi:")
input1 = st.number_input("Fitur 1", value=0.0)
input2 = st.number_input("Fitur 2", value=0.0)
input3 = st.number_input("Fitur 3", value=0.0)

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Mengambil data input pengguna
    input_data = [input1, input2, input3]
    
    # Prediksi
    result, epoch, bias, weights = predict(input_data)
    
    # Menampilkan hasil
    st.write(f"Hasil Prediksi: {result[0]}")
    st.write(f"Epoch: {epoch}")
    
    # Membuat tabel bias
    bias_table = pd.DataFrame({"Bias": bias})
    st.write("Tabel Bias:")
    st.table(bias_table)
    
    # Membuat tabel weights
    weights_table = pd.DataFrame(weights, columns=[f"Fitur {i+1}" for i in range(weights.shape[1])])
    st.write("Tabel Weights:")
    st.table(weights_table)
