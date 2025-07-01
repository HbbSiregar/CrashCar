import streamlit as st
import home, eda, predict

with st.sidebar:

    st.title("Project Deteksi Jenis Kerusakan Mobil")
    st.write('## Pilih Halaman')
    navigation = st.radio('Page', ['Home','EDA', 'Tools Prediksi Jenis Kerusakan Mobil'])
    
    st.write("# About")
    st.markdown('Proyek ini adalah proyek untuk pembuatan model deep learning dalam mendeteksi Jenis Kerusakan Mobil ') 

    
if navigation == "Home":
    home.home()
elif navigation == "EDA":
    eda.eda()
elif navigation == "Tools Prediksi Jenis Kerusakan Mobil" :
    predict.predict()


