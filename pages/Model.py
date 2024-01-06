import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import pad_sequences
from keras.models import load_model
from joblib import load

model = load('model_1.joblib')
st.header('Aplikasi Klasifikasi Keyword')  
model = load('model_1.joblib')
vocab = pickle.load(open('kbest_feature.pickle', 'rb'))

tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
 

options = ("Title", "Abstract")
selected_option = st.selectbox("Pilih Opsi:", options, index=None)

if selected_option == "Title":
   user_input = st.text_input('Masukkan Title :')
   tfidf = tf_idf_vec.fit_transform([user_input])
   prediksi = model.predict(tfidf)
   if st.button("prediksi"):
   
      if prediksi == 0:
         prediction = 'Exclude'
      elif prediksi == 1:
         prediction = 'Include'

      st.write("Hasil prediksi: adalah", prediction)

elif selected_option == "Abstract":
   user_input1 = st.text_input('Masukkan Abstract :')
   tfidf = tf_idf_vec.fit_transform([user_input1])
   prediksi = model.predict(tfidf)
   if st.button("prediksi"):
   
      if prediksi == 0:
         prediction = 'Exclude'
      elif prediksi == 1:
         prediction = 'Include'

      st.write("Hasil prediksi: adalah", prediction)