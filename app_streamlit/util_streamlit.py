##########################################
#####          LIBRERIAS             #####
##########################################

import streamlit as st
import geopandas as gpd
from data_processing import enc_cocina, enc_distrito, enc_barrio, encoder, mlb
import pandas as pd
from PIL import Image
import ast
import pickle
import folium
from streamlit_folium import st_folium
import random

##########################################
#####          FUNCIONES             #####
##########################################

''' 
Resumen:
- preferencias_culinarias_params(config, readme)
- servicios_params(config, readme)
- atcliente_params(config, readme)
- transformar_datos_user(df_user, barrios_mapa, barrios_m30)
- dicretizar_prediccion(pred)
- transformar_datos_empresa(barrio, df_usuario)
'''
#----------------------------------------------------------------------------------------------------------------------------------

def preferencias_culinarias_params(config, readme):
    ''' 
    Función para prefencias culinarias de Streamly.

    Input:
        config:str
        readme:str
    
    Output:
        seleccion_comida:str
        comida_vegetariana:bool
        vino:bool 
        cerveza:bool
    '''
    comida = ['Española', 'Latinoamericana', 'Italiana', 'China', 'Otros',
       'Japonesa', 'Americana / Burgers', 'Fusión', 'Mexicana',
       'Asiática']
    
    seleccion_comida = st.selectbox("Selecciona un tipo:", comida)
    comida_vegetariana = st.radio("¿Tienes menú vegetariano?", ["Sí", "No"])
    vino = st.radio("¿Tienes un carta de vinos?", ["Sí", "No"])
    cerveza = st.radio("¿Sirves Cerveza?", ["Sí", "No"])

    return seleccion_comida, comida_vegetariana, vino, cerveza

#----------------------------------------------------------------------------------------------------------------------------------

def servicios_params(config, readme):
    ''' 
    Función para servicios del restaurante de Streamly.

    Input:
        config:str
        readme:str
    
    Output:
        comer_dentro:bool
        acepta_reservas:bool
        takeout:bool 
        delivery:bool
        weelchair:bool
    '''
    comer_dentro = st.radio("¿Se puede comer en el local?", ["Sí", "No"])
    acepta_reservas = st.radio("¿Se puede hacer reservas?", ["Sí", "No"])
    takeout = st.radio("¿Pueden hacer pedidos para llevar?", ["Sí", "No"])
    delivery = st.radio("¿Vas a habilitar servicios de delivery?", ["Sí", "No"])
    weelchair = st.radio("¿Tu local será accesible para personas en silla de ruedas?", ["Sí", "No"])

    return comer_dentro, acepta_reservas, takeout, delivery, weelchair

#----------------------------------------------------------------------------------------------------------------------------------

def atcliente_params(config, readme):
    ''' 
    Función para atención del cliente en Streamlit.

    Input:
        config:str
        readme:str
    
    Output:
        serves_breakfast:bool
        serves_brunch:bool
        serves_dinner:bool 
        serves_lunch:bool
        open_weekends:bool
    '''
    serves_breakfast = st.radio("¿Servirán desayunos?", ["Sí", "No"])
    serves_brunch = st.radio("¿Servirán brunch?", ["Sí", "No"])
    serves_lunch = st.radio("¿Servirán comida?", ["Sí", "No"])
    serves_dinner = st.radio("¿Servirán cena?", ["Sí", "No"])
    open_weekends = st.radio("¿Abrirás los fines de semana?", ["Sí", "No"])
    
    return serves_breakfast, serves_brunch, serves_lunch, serves_dinner, open_weekends

#----------------------------------------------------------------------------------------------------------------------------------

def transformar_datos_user(df_user, barrios_mapa, barrios_m30):
    ''' 
    Funcion para formatear los datos del usuario para el modelo.

    Input:
        df_user:DataFrame
        barrios:GeoDataFrame
        barrios_m30:list
    
    Output:
        barrios:DataFrame
    '''
    barrios = barrios_mapa[barrios_mapa['COD_BAR'].isin(barrios_m30)]

    # Calcular centroides
    barrios_proj = barrios.to_crs(epsg=25830)
    barrios_proj['centroid'] = barrios_proj.geometry.centroid
    centroides_latlon = barrios_proj.set_geometry('centroid').to_crs(epsg=4326)
    barrios['lon'] = centroides_latlon.geometry.x
    barrios['lat'] = centroides_latlon.geometry.y
    barrios['centroid'] = centroides_latlon.geometry
    barrios = barrios[['CODDIS', 'NOMDIS', 'COD_BAR', 'NOMBRE', 'lon', 'lat', 'centroid']]

    # Restaurantes y buffers
    barrios_geo = gpd.GeoDataFrame(barrios, geometry='centroid', crs='EPSG:4326').to_crs(epsg=25830)
    barrios_geo['buffer_500'] = barrios_geo.centroid.buffer(500)
    barrios_geo = barrios_geo.set_geometry('buffer_500')

    restaurantes = pd.read_csv('../data/processed/restaurantes.csv')
    restaurantes = restaurantes[['lat', 'lon', 'place_id', 'price_level', 'rating',
                                 'user_ratings_total', 'anio_medio_constr_vivendas',
                                 'dur_media_credito_viviendas', 'edad_media_poblacion',
                                 'num_locales_alta_abiertos', 'num_locales_alta_cerrados',
                                 'poblacion_densidad', 'renta_media_persona',
                                 'pct_crecimiento_demografico', 'valor_catast_inmueble_residen',
                                 'tasa_parados', 'poblacion_80_mas', 'poblacion_italia',
                                 'poblacion_china', 'cod_barrio']]

    kpi = restaurantes.groupby('cod_barrio').max().reset_index()
    kpi.drop(['lat', 'lon', 'place_id', 'price_level', 'rating',
                                'user_ratings_total'], axis=1, inplace=True)

    restaurantes_geo = gpd.GeoDataFrame(
        restaurantes,
        geometry=gpd.points_from_xy(restaurantes['lon'], restaurantes['lat']),
        crs='EPSG:4326'
    ).to_crs(epsg=25830)

    result_restaurantes = gpd.sjoin(restaurantes_geo, barrios_geo, how='right', predicate='intersects')
    result = result_restaurantes.groupby(['COD_BAR'])[['price_level', 'rating', 'user_ratings_total']].mean().reset_index()
    result2 = result_restaurantes.groupby(['COD_BAR'])[['place_id']].count().reset_index()
    result = pd.merge(result, result2, on='COD_BAR')
    result.rename(columns={
        'price_level': 'price_level_mean',
        'rating': 'rating_mean',
        'user_ratings_total': 'user_ratings_mean',
        'place_id': 'num_restaurantes'
    }, inplace=True)

    barrios = pd.merge(barrios, result, how='left', on='COD_BAR')
    restaurantes.fillna(0, inplace=True)

    barrios['COD_BAR'] = barrios['COD_BAR'].astype(int)
    barrios['CODDIS'] = barrios['CODDIS'].astype(int)

    barrios = pd.merge(barrios, kpi, how='left', left_on='COD_BAR', right_on='cod_barrio')

    # Añadir valores del usuario
    for col in df_user.columns:
        valor = df_user.iloc[0][col]
        barrios[col] = [valor] * len(barrios) if isinstance(valor, list) else valor

    # ---- One Hot Encoding protegido ----
    if 'tipo_cocina' in barrios.columns and not barrios['tipo_cocina'].isnull().any():
        tip_coci = enc_cocina.transform(barrios[['tipo_cocina']]).toarray()
        tip_cocina_dummy = pd.DataFrame(tip_coci, columns=enc_cocina.categories_[0])
    else:
        tip_cocina_dummy = pd.DataFrame(0, index=barrios.index, columns=enc_cocina.categories_[0])

    if 'NOMDIS' in barrios.columns and not barrios['NOMDIS'].isnull().any():
        tip_distrito = enc_distrito.transform(barrios[['NOMDIS']]).toarray()
        tip_distrito_dummy = pd.DataFrame(tip_distrito, columns=enc_distrito.categories_[0])
    else:
        tip_distrito_dummy = pd.DataFrame(0, index=barrios.index, columns=enc_distrito.categories_[0])

    if 'NOMBRE' in barrios.columns and not barrios['NOMBRE'].isnull().any():
        tip_barrio = enc_barrio.transform(barrios[['NOMBRE']]).toarray()
        tip_barrio_dummy = pd.DataFrame(tip_barrio, columns=enc_barrio.categories_[0])
    else:
        tip_barrio_dummy = pd.DataFrame(0, index=barrios.index, columns=enc_barrio.categories_[0])

    barrios = pd.concat([barrios, tip_cocina_dummy, tip_distrito_dummy, tip_barrio_dummy], axis=1)

    # ---- MultiLabelBinarizer protegido ----
    def convertir_a_lista(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    def filtrar_etiquetas_validas(lista, clases_validas):
        lista_filtrada = []
        for sublista in lista:
            if isinstance(sublista, list):
                lista_filtrada.append([et for et in sublista if et in clases_validas])
            else:
                lista_filtrada.append([])
        return lista_filtrada

    barrios['types'] = barrios['types'].apply(convertir_a_lista)
    barrios['types'] = filtrar_etiquetas_validas(barrios['types'], mlb.classes_)

    if barrios['types'].apply(lambda x: isinstance(x, list) and len(x) > 0).any():
        types_encoded = mlb.transform(barrios['types'])
        types_df = pd.DataFrame(types_encoded, columns=mlb.classes_)
    else:
        types_df = pd.DataFrame(0, index=barrios.index, columns=mlb.classes_)

    barrios = pd.concat([barrios, types_df], axis=1)

    # ---- Limpieza final ----
    barrios.drop(['NOMDIS', 'NOMBRE', 'centroid', 'cod_barrio', 'tipo_cocina'], axis=1, inplace=True)
    barrios.rename(columns={'CODDIS': 'cod_distrito', 'COD_BAR': 'cod_barrio'}, inplace=True)

    barrios.fillna(0, inplace=True)

    barrios_full = barrios.copy()
    # Ordenamos columnas para el modelo
    barrios = barrios[['lat','lon','dine_in','price_level','reservable','serves_beer',
                                'serves_breakfast','serves_brunch','serves_dinner','serves_lunch',
                                'serves_vegetarian_food','serves_wine','takeout','delivery',
                                'weelchair','hours_open','num_days_open','open_weekends','amusement_park',
                                'bakery','bar','cafe','casino','convenience_store','grocery_or_supermarket',
                                'gym','health','liquor_store','lodging','meal_delivery','meal_takeaway',
                                'movie_theater','night_club','parking','real_estate_agency','spa',
                                'stadium','store','supermarket','tourist_attraction','travel_agency',
                                'price_level_mean','rating_mean','user_ratings_mean','num_restaurantes',
                                'anio_medio_constr_vivendas','dur_media_credito_viviendas','edad_media_poblacion',
                                'num_locales_alta_abiertos','num_locales_alta_cerrados','poblacion_densidad',
                                'renta_media_persona','pct_crecimiento_demografico','valor_catast_inmueble_residen',
                                'tasa_parados','poblacion_80_mas','poblacion_italia','poblacion_china','Americana / Burgers',
                                'Asiática','China','Española','Fusión','Italiana','Japonesa','Latinoamericana','Mexicana',
                                'Otros','Arganzuela','Carabanchel','Centro','Chamartín','Chamberí','Ciudad Lineal','Fuencarral - El Pardo',
                                'Hortaleza','Latina','Moncloa - Aravaca','Moratalaz','Puente de Vallecas','Retiro','Salamanca',
                                'Tetuán','Usera','Abrantes','Acacias','Adelfas','Almagro','Almenara','Almendrales','Aluche',
                                'Apóstol Santiago','Arapiles','Argüelles','Atalaya','Atocha','Bellas Vistas','Berruguete',
                                'Canillas','Casa de Campo','Castellana','Castilla','Castillejos','Chopera','Ciudad Jardín',
                                'Ciudad Universitaria','Colina','Comillas','Cortes','Costillares','Cuatro Caminos','Delicias',
                                'El Pardo','El Viso','Embajadores','Entrevías','Estrella','Fontarrón','Fuente del Berro',
                                'Fuentelarreina','Gaztambide','Goya','Guindalera','Hispanoamérica','Ibiza','Imperial',
                                'Justicia','La Concepción','La Paz','Legazpi','Lista','Los Cármenes','Los Jerónimos','Lucero',
                                'Marroquina','Media Legua','Mirasierra','Moscardó','Niño Jesús','Nueva España','Numancia',
                                'Opañel','Pacífico','Palacio','Palomeras Bajas','Palos de la Frontera','Peñagrande','Pilar',
                                'Pinar del Rey','Piovera','Portazgo','Pradolongo','Prosperidad','Puerta Bonita','Puerta del Ángel',
                                'Quintana','Recoletos','Ríos Rosas','San Diego','San Isidro','San Juan Bautista','San Pascual',
                                'Sol','Trafalgar','Universidad','Valdeacederas','Valdefuentes','Valdezarza','Vallehermoso',
                                'Valverde','Ventas','Vista Alegre','Zofío']]
    return barrios, barrios_full
#----------------------------------------------------------------------------------------------------------------------------------

def transformar_datos_empresa(barrio, df_usuario):
    ''' 
    Funcion para formatear los datos de la empresa para el modelo.

    Input:
        df_usuario:DataFrame
        barrio:GeoDataFrame
    
    Output:
        barrio_elegido:DataFrame
    '''
    barrio['COD_BAR'] = barrio['COD_BAR'].astype('int')
    barrio_elegido = barrio[barrio['COD_BAR']==int(df_usuario['cod_barrio'].values[0])]
    
    barrio_elegido = barrio_elegido.to_crs(epsg=25830)
    barrio_elegido['centroid'] = barrio_elegido.geometry.centroid # calculamos centroides de barrios
    centroides_latlon = barrio_elegido.set_geometry('centroid').to_crs(epsg=4326)
    barrio_elegido['lon'] = centroides_latlon.geometry.x
    barrio_elegido['lat'] = centroides_latlon.geometry.y
    barrio_elegido['centroid'] = centroides_latlon.geometry
    barrio_elegido = barrio_elegido[['CODDIS', 'NOMDIS', 'COD_BAR', 'NOMBRE','lon', 'lat', 'centroid']]

    # Para cada cada centroide calculamos un buffer de 500 para mirar que restaurantes tenemos cerca
    barrios_geo = gpd.GeoDataFrame(barrio_elegido, geometry='centroid', crs='EPSG:4326')
    barrios_geo = barrios_geo.to_crs(epsg=25830)
    barrios_geo['buffer_500'] = barrios_geo.centroid.buffer(500) #creamos el campo de buffer
    barrios_geo = barrios_geo.set_geometry('buffer_500')
    # leemos los restaurantes de madrid
    restaurantes = pd.read_csv('../data/processed/restaurantes.csv')
    restaurantes = restaurantes[['lat', 'lon', 'place_id', 'price_level', 'rating', 
                                'user_ratings_total', 'anio_medio_constr_vivendas',
                                'dur_media_credito_viviendas', 'edad_media_poblacion',
                                'num_locales_alta_abiertos', 'num_locales_alta_cerrados',
                                'poblacion_densidad', 'renta_media_persona',
                                'pct_crecimiento_demografico', 'valor_catast_inmueble_residen',
                                'tasa_parados', 'poblacion_80_mas',
                                'poblacion_italia', 'poblacion_china',
                                'cod_barrio']]

    # creamos un df de los kpi por barrio
    kpi = restaurantes.groupby('cod_barrio')[['anio_medio_constr_vivendas',
                                            'dur_media_credito_viviendas',
                                            'edad_media_poblacion',
                                            'num_locales_alta_abiertos',
                                            'num_locales_alta_cerrados',
                                            'poblacion_densidad',
                                            'renta_media_persona',
                                            'pct_crecimiento_demografico',
                                            'valor_catast_inmueble_residen',
                                            'tasa_parados',
                                            'poblacion_80_mas',
                                            'poblacion_italia',
                                            'poblacion_china']].max().reset_index()

    restaurantes_geo = gpd.GeoDataFrame(restaurantes, geometry=gpd.points_from_xy(restaurantes['lon'], restaurantes['lat']), crs='EPSG:4326')
    restaurantes_geo = restaurantes_geo.to_crs(epsg=25830)

    result_restaurantes = gpd.sjoin(restaurantes_geo, barrios_geo, how='right', predicate='intersects')
    result = result_restaurantes.groupby(['COD_BAR'])[['price_level', 'rating', 'user_ratings_total']].mean().reset_index()
    result2 = result_restaurantes.groupby(['COD_BAR'])[['place_id']].count().reset_index()
    result = pd.merge(result, result2, left_on=['COD_BAR'], right_on=['COD_BAR'])
    result.rename({
                'price_level':'price_level_mean',
                'rating':'rating_mean',
                'user_ratings_total':'user_ratings_mean',
                'place_id':'num_restaurantes'}, axis=1, inplace=True)

    # Resultado de buffer
    barrio_elegido = pd.merge(barrio_elegido, result, how='left', left_on='COD_BAR', right_on='COD_BAR')
    restaurantes.fillna(0, inplace=True)

    # Convertimos a int
    barrio_elegido['COD_BAR'] = barrio_elegido['COD_BAR'].astype('int')
    barrio_elegido['CODDIS'] = barrio_elegido['CODDIS'].astype('int')

    # Incluir los kpi a barrios
    barrio_elegido = pd.merge(barrio_elegido, kpi, how='left', left_on='COD_BAR', right_on='cod_barrio')

    # Incluimos datos del usuario
    for col in df_usuario.columns:
        barrio_elegido[col] = df_usuario.iloc[0][col]

    # One hot enconder
    tip_coci= enc_cocina.transform(barrio_elegido[['tipo_cocina']]).toarray()
    tip_cocina_dummy = pd.DataFrame(tip_coci, columns=[cat for cat in enc_cocina.categories_[0]])

    tip_distrito= enc_distrito.transform(barrio_elegido[['NOMDIS']]).toarray()
    tip_distrito_dummy = pd.DataFrame(tip_distrito, columns=[cat for cat in enc_distrito.categories_[0]])

    tip_barrio= enc_barrio.transform(barrio_elegido[['NOMBRE']]).toarray()
    tip_barrio_dummy = pd.DataFrame(tip_barrio, columns=[cat for cat in enc_barrio.categories_[0]])

    barrio_elegido = pd.concat([barrio_elegido, tip_cocina_dummy, tip_distrito_dummy, tip_barrio_dummy], axis=1)

    # MiltilabelBinarizer
    def convertir_a_lista(x):
        if isinstance(x, list):
            return x 
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    barrio_elegido['types'] = barrio_elegido['types'].apply(convertir_a_lista)

    types_encode = mlb.transform(barrio_elegido['types'])
    types_df = pd.DataFrame(types_encode, columns=mlb.classes_)

    barrio_elegido = pd.concat([barrio_elegido, types_df], axis=1) 

    # Quitamos Columnas
    barrio_elegido.drop(['NOMDIS', 'NOMBRE', 'centroid', 'cod_barrio', 'tipo_cocina'], axis=1, inplace=True)

    barrio_elegido.rename({'CODDIS':'cod_distrito',
                    'COD_BAR':'cod_barrio'}, inplace=True, axis=1)
    
    
    # Ordenamos columnas para el modelo
    barrio_elegido = barrio_elegido[['lat','lon','dine_in','price_level','reservable','serves_beer',
                                    'serves_breakfast','serves_brunch','serves_dinner','serves_lunch',
                                    'serves_vegetarian_food','serves_wine','takeout','delivery',
                                    'weelchair','hours_open','num_days_open','open_weekends','amusement_park',
                                    'bakery','bar','cafe','casino','convenience_store','grocery_or_supermarket',
                                    'gym','health','liquor_store','lodging','meal_delivery','meal_takeaway',
                                    'movie_theater','night_club','parking','real_estate_agency','spa',
                                    'stadium','store','supermarket','tourist_attraction','travel_agency',
                                    'price_level_mean','rating_mean','user_ratings_mean','num_restaurantes',
                                    'anio_medio_constr_vivendas','dur_media_credito_viviendas','edad_media_poblacion',
                                    'num_locales_alta_abiertos','num_locales_alta_cerrados','poblacion_densidad',
                                    'renta_media_persona','pct_crecimiento_demografico','valor_catast_inmueble_residen',
                                    'tasa_parados','poblacion_80_mas','poblacion_italia','poblacion_china','Americana / Burgers',
                                    'Asiática','China','Española','Fusión','Italiana','Japonesa','Latinoamericana','Mexicana',
                                    'Otros','Arganzuela','Carabanchel','Centro','Chamartín','Chamberí','Ciudad Lineal','Fuencarral - El Pardo',
                                    'Hortaleza','Latina','Moncloa - Aravaca','Moratalaz','Puente de Vallecas','Retiro','Salamanca',
                                    'Tetuán','Usera','Abrantes','Acacias','Adelfas','Almagro','Almenara','Almendrales','Aluche',
                                    'Apóstol Santiago','Arapiles','Argüelles','Atalaya','Atocha','Bellas Vistas','Berruguete',
                                    'Canillas','Casa de Campo','Castellana','Castilla','Castillejos','Chopera','Ciudad Jardín',
                                    'Ciudad Universitaria','Colina','Comillas','Cortes','Costillares','Cuatro Caminos','Delicias',
                                    'El Pardo','El Viso','Embajadores','Entrevías','Estrella','Fontarrón','Fuente del Berro',
                                    'Fuentelarreina','Gaztambide','Goya','Guindalera','Hispanoamérica','Ibiza','Imperial',
                                    'Justicia','La Concepción','La Paz','Legazpi','Lista','Los Cármenes','Los Jerónimos','Lucero',
                                    'Marroquina','Media Legua','Mirasierra','Moscardó','Niño Jesús','Nueva España','Numancia',
                                    'Opañel','Pacífico','Palacio','Palomeras Bajas','Palos de la Frontera','Peñagrande','Pilar',
                                    'Pinar del Rey','Piovera','Portazgo','Pradolongo','Prosperidad','Puerta Bonita','Puerta del Ángel',
                                    'Quintana','Recoletos','Ríos Rosas','San Diego','San Isidro','San Juan Bautista','San Pascual',
                                    'Sol','Trafalgar','Universidad','Valdeacederas','Valdefuentes','Valdezarza','Vallehermoso',
                                    'Valverde','Ventas','Vista Alegre','Zofío']]
    return barrio_elegido
#----------------------------------------------------------------------------------------------------------------------------------

def dicretizar_prediccion(pred):
    '''
    Funcion que discretiza la prediccion en categorias.

    Input:
        pred:float
    
    Output:
        rango_pred:str
    '''
    if pd.isna(pred):
        rango_pred = 'Sin predicción'
    if pred >= 40:
        rango_pred = 'Extremadamente popular'
    elif pred >= 30.8:
        rango_pred = 'Muy popular' 
    elif pred >= 26.5:
        rango_pred = 'Popular' 
    elif pred >= 20.85:
        rango_pred = 'Poco popular' 
    else:
        rango_pred = 'Muy poco popular'
    return rango_pred

#----------------------------------------------------------------------------------------------------------------------------------

def mostrar_formulario_empresas():
    ''' 
    Funcion para prediccion de empresas.
    '''
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

    .titulo-premium {
        font-size: 88px;
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        font-family: 'Poppins', sans-serif;
        color: #ee682e !important; 
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }

    .subtitulo {
        text-align: center;
        font-size: 24px;
        color: #555;
        margin-bottom: 40px;
        font-family: 'Poppins', sans-serif;
    }
    </style>

    <div class="titulo-premium">Madfood</div>
    <div class="subtitulo">Descubre la mejor ubicación para tu restaurante en el corazón de Madrid</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ¡Bienvenido a **Madfood**!

    Nos entusiasma acompañarte en la evaluación de riesgo de tu próximo restaurante.

    En **Madfood**, te ayudamos a validar si un restaurante ubicado dentro de la M-30 tiene potencial para ser popular.  
    Para brindarte la mejor recomendación posible, necesitamos conocer un poco más sobre tu idea.

    Cuéntanos los detalles, y te diremos si tu restaurante tiene lo que se necesita para destacar.

    ¡Vamos allá!
    """)

    
    # Header Pagina Principal ---------------------------------------------------------------------
    if st.sidebar.button("⬅️ Volver al menú principal"):
        st.session_state.modo = None
        st.experimental_rerun()
        st.stop()

    st.sidebar.image(Image.open('../doc/imagenes/Streamlit_logo.png'), use_container_width=True)

    st.sidebar.markdown("""Para calcular el riesgo del restaurante rellena este formulario.""")

    lista_barr = gpd.read_file('../data/raw/Barrios.json') # lectura
    lista_barr = gpd.GeoDataFrame(lista_barr, geometry='geometry', crs='EPSG:4326')
    codigos_barrios = [
        "011", "012", "013", "014", "015", "016",
        "021", "022", "023", "024", "025", "026", 
        "031", "032", "033", "034", "035", "036",
        "041", "042", "043", "044", "045", "046",
        "051", "052", "053", "074", "075","076",
        "061", "062", "063", "064", "065", "066",
        "071", "072", "073", "092", "054", "055",
        "081", "091", "121", "056", "085", "027",
        "131", "132", "133", "151", "093", "094",
        "084"
    ]
    lista = lista_barr[lista_barr['COD_BAR'].isin(codigos_barrios)][['NOMDIS', 'NOMBRE']]

    distritos = lista['NOMDIS'].unique()
    distrito_seleccionado = st.sidebar.selectbox("Selecciona un distrito", distritos)

    # Filtrar barrios según el distrito seleccionado
    barrios = lista[lista['NOMDIS'] == distrito_seleccionado]['NOMBRE'].unique()
    barrio_seleccionado = st.sidebar.selectbox("Selecciona un barrio", barrios)

    st.sidebar.write(f"Has seleccionado: **{distrito_seleccionado}** → **{barrio_seleccionado}**")

    fila_elegida = lista_barr[lista_barr['NOMBRE'] == barrio_seleccionado]
    valores = fila_elegida.iloc[0]
    cod_barrio = valores['COD_BAR']
    cod_distrito = valores['CODDIS']


    # Calculo Precio
    precio_opciones = [
        "1 - $ (Económico)", 
        "2 - $$ (Moderado)", 
        "3 - $$$ (Exclusivo)", 
        "4 - $$$$ (Premium)"
    ]

    precio_seleccionado = st.sidebar.radio(
        "Selecciona el rango de precio de tu restaurante:",
        options=precio_opciones
    )

    st.sidebar.write("Precio seleccionado:", precio_seleccionado)

    # Configuración de Opciones de Restaurante  ---------------------------------------------------
    st.sidebar.title("1. Preferencias Culinarias")

    with st.sidebar.expander("Tipo de Cocina", expanded=False):
        params_cocina = preferencias_culinarias_params({}, "")

    seleccion_comida, comida_vegetariana, vino, cerveza = params_cocina

    #---------------------------------------------------------------
    st.sidebar.title("2. Servicios")

    with st.sidebar.expander("Tipo de Servicios", expanded=False):
        params_servicios = servicios_params({}, "")

    comer_dentro, acepta_reservas, takeout, delivery, weelchair = params_servicios

    #---------------------------------------------------------------
    st.sidebar.title("3. Atención al Cliente")

    with st.sidebar.expander("Información de horarios", expanded=False):
        params_atcliente = atcliente_params({}, "")

    serves_breakfast, serves_brunch, serves_lunch, serves_dinner, open_weekends = params_atcliente
    #---------------------------------------------------------------

    with st.sidebar.expander("Distribución de horarios", expanded=False):
        horas_abierto = st.slider("¿Cuántas horas a la semana estará abierto?", 1, 150, 80)
        dias_abierto = st.slider("¿Cuántos días a la semana estará abierto?", 1, 7, 5)

    #---------------------------------------------------------------

    etiquetas_posibles = mlb.classes_

    etiquetas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las etiquetas que describen tu restaurante:",
    options=etiquetas_posibles,
    default=['bar']
    )
    entrada_types = [etiquetas_seleccionadas]

    #------------------------- Data Frame Resultados Usuario ------------------------------------
    data_usuario = {
        "price_level": [float(precio_seleccionado[0])],
        "tipo_cocina":[seleccion_comida],
        "serves_vegetarian_food":[int(bool(comida_vegetariana))],
        "serves_wine":[int(bool(vino))],
        "serves_beer":[int(bool(cerveza))],
        "dine_in":[int(bool(comer_dentro))],
        "reservable":[int(bool(acepta_reservas))],
        "takeout":[int(bool(takeout))],
        "delivery":[int(bool(delivery))],
        "weelchair":[int(bool(weelchair))],
        "serves_breakfast": [int(bool(serves_breakfast))],
        "serves_brunch": [int(bool(serves_brunch))],
        "serves_lunch": [int(bool(serves_lunch))],
        "serves_dinner": [int(bool(serves_dinner))],
        "open_weekends": [int(bool(open_weekends))],
        "hours_open": [float(horas_abierto)],
        "num_days_open": [int(dias_abierto)],
        'cod_distrito': [int(cod_distrito)],
        'cod_barrio': [int(cod_barrio)],
        'types':[entrada_types]
    }

    df_usuario = pd.DataFrame(data_usuario)

    barrios = transformar_datos_empresa(lista_barr, df_usuario)


    modelo_importado = pickle.load(open("../models/final_model.pkl", 'rb'))

    #---------------------------------- Mapa --------------------------------------
    if "prediccion" not in st.session_state:
        st.session_state.prediccion = None

    if st.sidebar.button("Predecir"):
        st.session_state.prediccion = modelo_importado.predict(barrios)
        st.success("✅ Predicción realizada")
    
    if st.session_state.prediccion is not None:

        barrios["rating_predicho"] = st.session_state.prediccion
        barrios["rating_discreto"] = barrios["rating_predicho"].apply(dicretizar_prediccion)

        res = pd.read_csv('../data/processed/restaurantes.csv')

        res = res[res['cod_barrio']==df_usuario['cod_barrio'].values[0]]


        res_df = res.groupby('cod_barrio')[['price_level', 'rating', 'user_ratings_total']].mean().reset_index()

        res_df = res_df.round(2)

        res_df.rename({'price_level':'Media Barrio Precio', 
                       'rating':'Media de Barrio Rating', 
                       'user_ratings_total':'Media Barrio Num Reviews'}, axis=1, inplace=True)

        result = pd.concat([barrios, res_df], axis=1)

        result = result[['Media Barrio Precio', 'Media de Barrio Rating', 'Media Barrio Num Reviews', 'rating_discreto']]

        result.rename({'rating_discreto':'Predicción Popularidad'}, axis=1, inplace=True)


        st.dataframe(result, hide_index=True)
    
        mapa_simple = folium.Map(location=[float(barrios['lat'].values[0]), float(barrios['lon'].values[0])], 
                                tiles="OpenStreetMap", zoom_start=14.5)

        for lat, lon, nom in zip(res['lat'], res['lon'], res['nombre_restaurante']):
            folium.Marker([lat, lon], 
                popup= nom, 
                icon=folium.Icon(icon='cutlery', prefix='fa')).add_to(mapa_simple)

        st.write(f"**Mapa de {barrio_seleccionado} Madrid**")
        st.write(f"Este mapa contiene los restaurantes de la zona.")
        st_folium(mapa_simple, width=700)

        
        

#----------------------------------------------------------------------------------------------------------------------------------
def mostrar_formulario_user():
    ''' 
    Funcion para prediccion de autonomos.
    '''
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

    .titulo-premium {
        font-size: 88px;
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        font-family: 'Poppins', sans-serif;
        color: #ee682e !important; 
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }

    .subtitulo {
        text-align: center;
        font-size: 24px;
        color: #555;
        margin-bottom: 40px;
        font-family: 'Poppins', sans-serif;
    }
    </style>

    <div class="titulo-premium">Madfood</div>
    <div class="subtitulo">Descubre la mejor ubicación para tu restaurante en el corazón de Madrid</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ¡Bienvenido a **Madfood**!

    Nos emociona acompañarte en el emocionante camino de abrir tu nuevo restaurante.

    Nuestro objetivo es ayudarte a encontrar la ubicación ideal para tu negocio dentro de la M30, el corazón gastronómico de Madrid.  
    Para hacerlo de la mejor manera, necesitamos que nos cuentes un poco más sobre tu concepto e ideas.

    Con esa información, te recomendaremos el barrio que mejor se alinee con el estilo y las características de tu restaurante, maximizando tus oportunidades de éxito.

    ¡Empecemos a diseñar tu futuro juntos!
    """)

    
    # Header Pagina Principal ---------------------------------------------------------------------
    if st.sidebar.button("⬅️ Volver al menú principal"):
        st.session_state.modo = None
        st.experimental_rerun()
        st.stop()

    st.sidebar.image(Image.open('../doc/imagenes/Streamlit_logo.png'), use_container_width=True)

    st.sidebar.markdown("""Para calcular el exito de tu futuro restaurante rellena este formulario,
            tranquila/o no te voy a robar la idea. 😆""")
    
    # Calculo Precio
    precio_opciones = [
        "1 - $ (Económico)", 
        "2 - $$ (Moderado)", 
        "3 - $$$ (Exclusivo)", 
        "4 - $$$$ (Premium)"
    ]

    precio_seleccionado = st.sidebar.radio(
        "Selecciona el rango de precio de tu restaurante:",
        options=precio_opciones
    )

    st.sidebar.write("Precio seleccionado:", precio_seleccionado)

    # Configuración de Opciones de Restaurante  ---------------------------------------------------
    st.sidebar.title("1. Preferencias Culinarias")

    with st.sidebar.expander("Tipo de Cocina", expanded=False):
        params_cocina = preferencias_culinarias_params({}, "")

    seleccion_comida, comida_vegetariana, vino, cerveza = params_cocina

    #---------------------------------------------------------------
    st.sidebar.title("2. Servicios")

    with st.sidebar.expander("Tipo de Servicios", expanded=False):
        params_servicios = servicios_params({}, "")

    comer_dentro, acepta_reservas, takeout, delivery, weelchair = params_servicios

    #---------------------------------------------------------------
    st.sidebar.title("3. Atención al Cliente")

    with st.sidebar.expander("Información de horarios", expanded=False):
        params_atcliente = atcliente_params({}, "")

    serves_breakfast, serves_brunch, serves_lunch, serves_dinner, open_weekends = params_atcliente
    #---------------------------------------------------------------

    with st.sidebar.expander("Distribución de horarios", expanded=False):
        horas_abierto = st.slider("¿Cuántas horas a la semana estará abierto?", 1, 150, 80)
        dias_abierto = st.slider("¿Cuántos días a la semana estará abierto?", 1, 7, 5)

    #---------------------------------------------------------------

    etiquetas_posibles = mlb.classes_

    etiquetas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las etiquetas que describen tu restaurante:",
    options=etiquetas_posibles,
    default=['bar']
    )
    entrada_types = [etiquetas_seleccionadas]

    #------------------------- Data Frame Resultados Usuario ------------------------------------
    data_usuario = {
        "price_level": [float(precio_seleccionado[0])],
        "tipo_cocina":[seleccion_comida],
        "serves_vegetarian_food":[int(bool(comida_vegetariana))],
        "serves_wine":[int(bool(vino))],
        "serves_beer":[int(bool(cerveza))],
        "dine_in":[int(bool(comer_dentro))],
        "reservable":[int(bool(acepta_reservas))],
        "takeout":[int(bool(takeout))],
        "delivery":[int(bool(delivery))],
        "weelchair":[int(bool(weelchair))],
        "serves_breakfast": [int(bool(serves_breakfast))],
        "serves_brunch": [int(bool(serves_brunch))],
        "serves_lunch": [int(bool(serves_lunch))],
        "serves_dinner": [int(bool(serves_dinner))],
        "open_weekends": [int(bool(open_weekends))],
        "hours_open": [float(horas_abierto)],
        "num_days_open": [int(dias_abierto)],
        'types':entrada_types
    }

    df_usuario = pd.DataFrame(data_usuario)

    barrios_mapa = gpd.read_file('../data/raw/Barrios.json')
    barrios_mapa = gpd.GeoDataFrame(barrios_mapa, geometry='geometry', crs='EPSG:4326') 

    #---------------------------------------------------------------
    # Modelo

    codigos_barrios = [
        "011", "012", "013", "014", "015", "016",
        "021", "022", "023", "024", "025", "026", 
        "031", "032", "033", "034", "035", "036",
        "041", "042", "043", "044", "045", "046",
        "051", "052", "053", "074", "075","076",
        "061", "062", "063", "064", "065", "066",
        "071", "072", "073", "092", "054", "055",
        "081", "091", "121", "056", "085", "027",
        "131", "132", "133", "151", "093", "094",
        "084"
    ]

    barrios, barrios_full = transformar_datos_user(df_usuario, barrios_mapa, codigos_barrios)

    modelo_importado = pickle.load(open("../models/final_model.pkl", 'rb'))

    #---------------------------------- Mapa --------------------------------------
    if "prediccion" not in st.session_state:
        st.session_state.prediccion = None

    if st.sidebar.button("Predecir"):
        st.session_state.prediccion = modelo_importado.predict(barrios)
        st.success("✅ Predicción realizada")
        
    # ---------------------------------------
    # Mostrar mapa según predicción

    if st.session_state.prediccion is not None:

        barrios["rating_predicho"] = st.session_state.prediccion
        barrios["rating_discreto"] = barrios["rating_predicho"].apply(dicretizar_prediccion)

        barrios['cod_barrio'] = barrios_full['cod_barrio']

        barrios_mapa['COD_BAR'] = barrios_mapa['COD_BAR'].astype(int)
        barrios_pred = pd.merge(barrios_mapa, barrios[['cod_barrio', 'rating_discreto', 'rating_predicho']],
                                left_on='COD_BAR', right_on='cod_barrio', how='left')
        st.write("📊 **Top 5 Barrios con mayor rating predicho**")

        barrios_pred_copy = barrios_pred.copy()
        barrios_pred_copy.rename(columns={'NOMBRE':'Barrio',
                                'rating_discreto':'Rating',
                                'NOMDIS': 'Distrito'}, inplace=True)
        top = barrios_pred_copy[['Distrito', 'Barrio', 'Rating', 'rating_predicho']].sort_values(
            by='rating_predicho', ascending=False).reset_index(drop=True)
        
        top_5 = top[['Distrito', 'Barrio', 'Rating']].head(5)

        st.markdown("""
        Ahora que tenemos el top 5 de barrios puedes empezar a buscar el mejor local para tu restaurante:

        * [Alquilar Local](https://www.idealista.com/alquiler-locales/madrid-madrid/con-publicado_ultima-semana,restauracion/)
        * [Comprar Local](https://www.idealista.com/venta-locales/madrid-madrid/con-publicado_ultima-semana,restauracion/)
        """)

        st.dataframe(top_5, hide_index=True)

        kpi_barrios = pd.read_csv('../data/raw/kpi_barrios_madrid.csv')
        
        barrio_elegido = random.choice(top_5['Barrio'])
        
        kpi_barrios = kpi_barrios[kpi_barrios['barrio']==barrio_elegido]

        kpi_elegido = random.choice(kpi_barrios['indicador_completo'].tolist())

        kpi_barrios = kpi_barrios[kpi_barrios['indicador_completo']==kpi_elegido]

        valor_elegido = float(kpi_barrios['valor_indicador'].values[0])

        valor_formateado = f"{valor_elegido:,.0f}".replace(",", ".")

        st.write(f"Fun fact del barrio {barrio_elegido}: {kpi_elegido} es de {valor_formateado}.")


        m = folium.Map(location=[40.4165, -3.70256], zoom_start=12, tiles="Cartodb Positron")

        folium.Choropleth(
            geo_data=barrios_pred,
            data=barrios_pred,
            columns=['COD_BAR', 'rating_predicho'],
            key_on='feature.properties.COD_BAR',
            fill_color='BuPu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Rating Predicho por Barrio'
        ).add_to(m)

        for _, r in barrios_pred.iterrows():
            folium.GeoJson(
                r['geometry'],
                tooltip=f"{r['NOMBRE']} - Rating predicho: {r['rating_discreto']}"
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st.write("🗺️ **Mapa de Rating Predicho por Barrio**")
        st_folium(m, width=700)

        
    else:
        barrios_mapa = gpd.read_file('../data/raw/Barrios.json')
        barrios_mapa = gpd.GeoDataFrame(barrios_mapa, geometry='geometry', crs='EPSG:4326') 

        mapa_simple = folium.Map(location=[40.4165, -3.70256], tiles="OpenStreetMap", zoom_start=12.5)

        for _, row in barrios_mapa.iterrows():
            folium.GeoJson(row['geometry'],
                        tooltip=row['NOMBRE']).add_to(mapa_simple)

        st.write("🗺️ **Mapa de Barrios de Madrid (sin predicción todavía)**")
        st_folium(mapa_simple, width=700)


