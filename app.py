import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import pad_sequences
from keras.models import load_model
from joblib import load


st.image('image/a.jpg')
st.title('Automatic Systematic Literature Review')
st.write("""Selamat datang di Automatic Systematic Literature Review. Aplikasi ini menggunakan
         teknologi deep learning untuk mengidentifikasi berbagai kata dari abstrack dan title. 
         Dengan menggunakan algoritma klasifikasi canggih, aplikasi ini dapat 
         membantu dokter dan peneliti dalam mengklasifikasikan title dan abstract .""")

st.header("Cara Penggunaan")
st.text("1. Masukkan keyword untuk melakukan prediksi")
st.text("2. Click tombol 'Prediksi'")
st.text("3. Tunggu hasil prediksi dan lihat hasil predksi")

st.header("Metode dan Tujuan dari website ini", divider="gray")
col1, col2 = st.columns(2)
with col1:
   st.subheader("Teknologi Ai")
   st.write("""Aplikasi ini menggunakan teknologi berbasi AI
             dengan menggunakan metode deep learning dan
            feature representation tfidf""")
   
with col2:
   st.subheader("Screening Artikel")
   st.write("""Aplikasi ini dapat membantu para dokter dan 
            peneliti untuk membantu screening artikel dengan efisien""")
    


