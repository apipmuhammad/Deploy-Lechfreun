import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import pad_sequences
from keras.models import load_model
from joblib import load

st.header('Our Teams')
st.write("""Dalam proses pembuatan website aplikasi klasifikasi keyword 
         disini kami terdiri dari 3 orang""")

col1, col2, col3 = st.columns(3)
with col1:
   original_title = '<p font-size: 40px;"></p>'
   st.markdown(original_title, unsafe_allow_html=True)
   st.image('image/Picture1.png', caption='Dikco Agung Prasetyo')

with col2:
   original_title = '<p font-size: 40px; "></p>'
   st.markdown(original_title, unsafe_allow_html=True)
   st.image('image/Picture2.png', caption='Muhamad Ivan Fadhillah')

with col3:
   original_title = '<p font-size: 40px;"></p>'
   st.markdown(original_title, unsafe_allow_html=True)
   st.image('image/Picture3.png', caption='Muhammad Soleh Apriadi')
