import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import os

def home():
    # Judul tengah & responsif
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='font-size: 2.2em;'>KLASIFIKASI TINGKAT KERUSAKAN MOBIL PASCA KECELAKAAN</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Gambar banner
    image_url = 'src/Banner.png'
    gambar = Image.open(image_url)
    st.image(gambar, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Latar Belakang
    st.markdown("## Latar Belakang")
    st.markdown(
        """
        <div style='text-align: justify;'>
        Dalam industri asuransi, proses inspeksi kerusakan kendaraaan oleh petugas inspeksi masih dilakukan secara manual. 
        Proses itu memerlukan waktu yang relatif lama dan ketidakakuratan karena subjektifitas surveyor, dimana hal-hal ini bisa menurunkan kepuasan pelanggan. 
        Untuk mengatasi masalah tersebut diperlukan sistem otomatis yang dapat memprediksi klasifikasi tingkat kerusakan mobil. 
        Jadi kerusakan itu bisa terdeteksi dari gambar yang diunggah oleh pemilik mobil.
        Sistem ini berguna untuk mengotomasi proses inspeksi sehingga dapat mempercepat pengambilan keputusan, mengurangi beban petugas inspeksi asuransi dan meningkatkan akurasi penilaian pada kerusakan. 
        Pengguna sistem ini adalah si pemilik mobil yang mengajukan klaim asuransi dan petugas inspeksi asuransi.
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Problem Statement
    st.markdown("## Problem Statement")
    st.markdown(
        """
        <div style='text-align: justify;'>
        Pembuatan model klasifikasi menggunakan deep learning untuk mengelompokkan tingkat kerusakan mobil akibat kecelakaan ke dalam tiga kategori, yaitu:
        <ul>
            <li><b>Minor:</b> Kerusakan ringan pada mobil, seperti goresan kecil atau penyok ringan.</li>
            <li><b>Moderate:</b> Kerusakan sedang yang membutuhkan perbaikan struktural sebagian.</li>
            <li><b>Severe:</b> Kerusakan berat yang umumnya membuat mobil tidak dapat dikendarai dan membutuhkan perbaikan besar atau penggantian total.</li>
        </ul>
        Model ini diharapkan dapat membantu dalam penilaian otomatis dan efisien terhadap kondisi kendaraan pasca kecelakaan.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dataset
    st.markdown("## Dataset")
    st.markdown(
        """
        Dataset yang digunakan berasal dari platform <b>Kaggle</b>. Dataset ini berisi berbagai gambar dari mobil pasca kecelakaan,
        dan diklasifikasikan ke dalam tiga kelas: <b>Minor</b>, <b>Moderate</b>, dan <b>Severe</b>.
        """, 
        unsafe_allow_html=True
    )

    # Contoh gambar
    st.markdown("### Contoh Gambar dari Masing-masing Kelas")

    col1, col2, col3 = st.columns(3)

    kelas_folder = {
        "Minor": "src/data3a/training/01-minor",
        "Moderate": "src/data3a/training/02-moderate",
        "Severe": "src/data3a/training/03-severe"
    }

    for (label, folder), col in zip(kelas_folder.items(), [col1, col2, col3]):
        try:
            gambar_list = os.listdir(folder)
            if gambar_list:
                gambar_path = os.path.join(folder, gambar_list[0])  # Gambar pertama
                img = Image.open(gambar_path)
                col.image(img, caption=label, use_container_width=True)
            else:
                col.markdown(f"**Tidak ada gambar di folder {label}**")
        except Exception as e:
            col.markdown(f"**Gagal memuat gambar {label}: {e}**")

    # Tabel Keterangan
    st.markdown("## Keterangan Kolom")
    st.markdown(
        """
        <div style='text-align: justify;'>
        Berikut ini adalah beberapa fitur atau label target yang digunakan untuk membedakan tingkat kerusakan mobil:
        </div>
        """, 
        unsafe_allow_html=True
    )

    data = {
        "Nama Kolom": [
            "01-minor", "02-moderate", "03-severe"
        ],
        "Keterangan": [
            "Kerusakan ringan: goresan, penyok kecil",
            "Kerusakan sedang: perbaikan struktural sebagian",
            "Kerusakan berat: tidak layak jalan, perlu perbaikan besar"
        ]
    }

    df_keterangan = pd.DataFrame(data)
    st.table(df_keterangan)

if __name__ == '__main__':
    home()
