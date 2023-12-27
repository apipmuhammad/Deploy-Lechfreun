import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.models import Model
from keras.preprocessing.text import Tokenizer
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



# Memuat model 
model = load('model_1.joblib')

# Memuat fitur seleksi (TfidfVectorizer) jika diperlukan
#vectorizer = pickle.load(open("new_tfidf.pickle", "rb"))


# Antarmuka pengguna Streamlit
st.header('Aplikasi Klasifikasi Keyword')

# Input keyword dari pengguna

#txts = tok.texts_to_sequences([user_input])
#txts = pad_sequences(txts)
#preds = model.predict(txts)

#if st.button('Predict'):
    #if(preds == 0):
       #kelas = "Paper ini termasuk kategori exclude"
    #else:
       #kelas = "Paper ini termasuk kategori include"
model = load('model_1.joblib')
vocab = pickle.load(open('kbest_feature.pickle', 'rb'))


tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
user_input = st.text_input('Masukkan sebuah keyword:')  
tfidf = tf_idf_vec.fit_transform([user_input])
prediksi = model.predict(tfidf)
if st.button("prediksi"):
    st.subheader("prediksi:")
    if prediksi == 0:
        prediction = 'Keyword diklasifikasikan sebagai 0 '
    elif prediksi == 1:
        prediction = 'Keyword diklasifikasikan sebagai 1.'

    print('Hasil prediksi', user_input, ' adalah', prediction)
    
#if st.button("Prediksi"):  
    # Praolah input menggunakan vectorizer yang telah dimuat
    #input_vectorized = tf_idf_vec.fit_transform([user_input])
    
    # Melakukan prediksi menggunakan model yang telah dimuat
   

    #result = ''
    # Menampilkan hasil
    #st.subheader('Prediksi:')
    #if prediksi == 0:
        #result = 'Keyword diklasifikasikan sebagai 0 .'
    #else:
        #result = 'Keyword diklasifikasikan sebagai 1.'

    #print('Hasil prediksi', user_input, ' adalah', result)
   
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

