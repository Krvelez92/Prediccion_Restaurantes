##########################################
#####          FUNCIONES             #####
##########################################


import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from PIL import Image
import util_streamlit as u
import pandas as pd
import pickle
import random


# ----------------- Configuración Pagina Tab ---------------------

apptitle = 'MadFood'

st.set_page_config(page_title=apptitle, page_icon=":ramen:")

if "modo" not in st.session_state:
    st.session_state.modo = None

if st.session_state.modo is None:
    st.markdown("""
    <style>
    .header-banner {
        background-image: url("https://raw.githubusercontent.com/Krvelez92/Prediccion_Restaurantes/main/doc/imagenes/header.png");
        background-repeat: repeat-x;
        background-position: top center;
        background-size: auto 150px; 
        height: 150px;
    }
    </style>

    <div class="header-banner"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

        .titulo-premium {
            font-size: 88px;
            font-weight: 800;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 15px;
            font-family: 'Poppins', sans-serif;
            color: #ee682e !important; 
            text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
        }

        .subtitulo {
            text-align: center;
            font-size: 24px;
            color: #555;
            margin-bottom: 30px;
            font-family: 'Poppins', sans-serif;
        }
        </style>

        <div class="titulo-premium">Madfood</div>
        <div class="subtitulo">Descubre la mejor ubicación para tu restaurante en el corazón de Madrid</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ¡Bienvenido a **Madfood**!

    Nos emociona acompañarte en el emocionante camino del mundo de restauración.
    ¡Empecemos a diseñar tu futuro juntos!
    """)
    st.markdown("### ¿Qué quieres explorar?")
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👔 Popularidad Restaurante"):
            st.session_state.modo = "empresas"

    with col2:
        if st.button("🧑‍🔧 ¿Donde soy popular?"):
            st.session_state.modo = "autonomos"

    st.markdown("""
    <style>
        .stButton > button {
            font-size: 20px;
            padding: 0.75em 2em;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)
    

if st.session_state.modo == "empresas":
    st.success("🔎 Modo seleccionado: Empresas")
    u.mostrar_formulario_empresas()

elif st.session_state.modo == "autonomos":
    st.success("🔎 Modo seleccionado: Autónomos")
    u.mostrar_formulario_user()