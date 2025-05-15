##########################################
#####          LIBRERIAS             #####
##########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils as u
import re
import geopandas as gpd
import ast
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#----------------------------------------------------------------------------------------------------------------------------------

''' 
Lectura de datos de id de sitios y sus nombres.
'''

restaurantes = pd.read_csv('../data/raw/lugares_madrid.csv')

''' 
Lectura de datos de detalle de sitios de Madrid.
'''
detalle_restaurantes = pd.read_csv('../data/raw/detalle_sitios.csv')

restaurantes = pd.merge(restaurantes, detalle_restaurantes, left_on='id', right_on='place_id') # Unimos las 2 fuentes por el place_id

restaurantes.drop(['id', 'summary'], inplace=True, axis=1) # Quitamos las  variables de id y summary de dataframe
restaurantes = restaurantes[restaurantes['rating'].notnull()] # Quitamos los restaurantes que tienen rating vacios

#----------------------------------------------------------------------------------------------------------------------------------

''' 
Nos quitamos registros que en google maps estan marcados con restaurantes pero no lo son 
'''

tipos = ['beauty_salon', 'clothing_store', 'doctor', 'shopping_mall', 'physiotherapist']

for i in tipos:
    valor = restaurantes[restaurantes['types'].str.contains(i)].index
    restaurantes.drop(index=valor, axis=0, inplace=True)
    restaurantes.reset_index(drop=True, inplace=True)

''' 
Ahora que hemos quitado los registros vamos a limpiar un poco la variable types para quedarnos con las categorias
que importan y no tan generales.
'''

rem = ['food', 'point_of_interest', 'establishment', 'restaurant', 'point_of_interest']

for i, j in enumerate(restaurantes['types']):
    list = ast.literal_eval(j)
    lista_filtrada = [k for k in list if k not in rem]
    if len(lista_filtrada)>0:
        restaurantes.at[i, 'types'] = lista_filtrada
    else:
        restaurantes.at[i, 'types'] = None

restaurantes_sinnan = restaurantes[restaurantes['types'].notna()].reset_index(drop=True) #Creamos un df sin tipos none

''' 
Vamos a crear una columna por cada categoria y marcamos con 1 si tiene esa etiqueta.
'''

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
types_encode = mlb.fit_transform(restaurantes_sinnan['types'])
types_df = pd.DataFrame(types_encode, columns=mlb.classes_)

restaurantes_sinnan = pd.concat([restaurantes_sinnan, types_df], axis=1) 

restaurantes = pd.concat([restaurantes_sinnan, restaurantes[restaurantes['types'].isna()]], axis=0)
restaurantes.drop(columns='types', inplace=True)

for i in types_df.columns:
    restaurantes[i] = restaurantes[i].fillna(0) #Rellenamos los tipos que vienen vacios con 0 


#----------------------------------------------------------------------------------------------------------------------------------

''' 
Vamos hacer un proceso de limpieza para el dataframe y vamos a asumir cierto valores para los registros sin datos.
- todas las variables booleanas según cada un asumiremos que los valores vacios son verdaderos o falsos.
- dine_in: Comer dentro del restaurante todos los vacios asumimos que son True (lo más común).
- reservable: Si se puedo o no hacer reservas es dificil determinar su valor pero asumimos que es Falso. 
- serves_beer: Si sirve cerveza, asumimos que siempre. 
- serves_breakfast: Sirven desayunos, asumimos que no ya que lo común es que sea almuerzos o cena. 
- serves_brunch:Sirven brunch, asumimos que no ya que lo común es que sea almuerzos o cena. 
- serves_dinner: Sirven cena, asumimos que si ya que lo común es que sea almuerzos o cena. 
- serves_lunch: Sirven almuerzo, asumimos que si ya que lo común es que sea almuerzos o cena. 
- serves_vegetarian_food: Sirven comida vegetariana, asumimos que no ya que no es aun tan común aquí en Madrid. 
- serves_wine: Si sirve vino, asumimos que siempre.
- takeout: Si sirven para llevar asumimos que si suelen permitir.
- delivery: Asumimos que no hacen delivery.
- weelchair: Asumimos que no tienen acceso para sillas de ruedas.
- hours_open: Asumimos que el horario de apertura es el más común.
'''
moda_open = restaurantes['hours_open'].mode()[0]

restaurantes['price_level'] = restaurantes['price_level'].fillna(1)
restaurantes['dine_in'] = restaurantes['dine_in'].astype(bool).fillna(True).astype(int)
restaurantes['reservable'] = restaurantes['reservable'].astype(bool).fillna(False).astype(int)
restaurantes['serves_beer'] = restaurantes['serves_beer'].astype(bool).fillna(True).astype(int)
restaurantes['serves_breakfast'] = restaurantes['serves_breakfast'].astype(bool).fillna(False).astype(int)
restaurantes['serves_brunch'] = restaurantes['serves_brunch'].astype(bool).fillna(False).astype(int)
restaurantes['serves_dinner'] = restaurantes['serves_dinner'].astype(bool).fillna(True).astype(int)
restaurantes['serves_lunch'] = restaurantes['serves_lunch'].astype(bool).fillna(True).astype(int)
restaurantes['serves_vegetarian_food'] = restaurantes['serves_vegetarian_food'].astype(bool).fillna(False).astype(int)
restaurantes['serves_wine'] = restaurantes['serves_wine'].astype(bool).fillna(True).astype(int)
restaurantes['takeout'] = restaurantes['takeout'].astype(bool).fillna(True).astype(int)
restaurantes['delivery'] = restaurantes['delivery'].astype(bool).fillna(False).astype(int)
restaurantes['weelchair'] = restaurantes['weelchair'].astype(bool).fillna(False).astype(int)
restaurantes['num_days_open'] = np.where(restaurantes['num_days_open']==0, 7, restaurantes['num_days_open'])
restaurantes['hours_open'] = restaurantes['hours_open'].fillna(moda_open)
restaurantes['open_weekends'] = restaurantes['open_weekends'].astype(int)

#----------------------------------------------------------------------------------------------------------------------------------
##########################################
#####          GEOPANDAS             #####
##########################################


''' 
Creamos un dataframe geoespacial donde los puntos de geometria es la latitud y logitud. Esta medidas tiene un valor de crs 
(Coordinate Reference System of the geometry objects) de 4326 que hace referencia a la latitud y logitud.
'''

restaurantes_geo = gpd.GeoDataFrame(restaurantes, geometry=gpd.points_from_xy(restaurantes['lon'], restaurantes['lat']), crs='EPSG:4326')

''' 
Lectura de geometria de barrios y las transformamos en geopandas con un crs de 4326 para poder mapear con nuestro dataframe de restaurantes_geo.
'''
barrios = gpd.read_file('../data/raw/Barrios.json') # lectura
barrios = gpd.GeoDataFrame(barrios, geometry='geometry', crs='EPSG:4326') # Determinar la geometria

'''
Una vez unficado las 2 fuentes en la misma métricas vamos a hacer un join espacial donde miramos de la geometria de los restaurantes en que barrio
se encuentra dentro o intersecan.
Quitamos variables que no necesitamos.
'''

restaurantes_geo = gpd.sjoin(restaurantes_geo, barrios, how='left', predicate='intersects')
restaurantes_geo.drop(['index_right', 'id', 'Shape_Leng', 'Shape_Area','FCH_ALTA', 
                       'FCH_BAJA', 'OBSERVACIO', 'APROBACION', 'COD_DIS_TX', 'COD_DISB',
                       'BARRIO_MAY','BARRIO_MT', 'COD_DISBAR', 'NUM_BAR'], axis=1 , inplace=True)

restaurantes = pd.merge(restaurantes, restaurantes_geo[['place_id', 'CODDIS', 'NOMDIS','COD_BAR', 'NOMBRE']], 
                        how='left', left_on='place_id', right_on='place_id')

#----------------------------------------------------------------------------------------------------------------------------------

restaurantes_geo = restaurantes_geo.to_crs(epsg=25830) #cambiar lat y log a seudonimo en distancia en metros para España

''' 
Creamos una copia del dataframe de restarantes para crearlo con el buffer de los 500 m.
'''
restaurantes_geo_buffer = restaurantes_geo.copy() #copia de df
restaurantes_geo_buffer['buffer_500'] = restaurantes_geo_buffer.geometry.buffer(500) #creamos el campo de buffer
restaurantes_geo_buffer = restaurantes_geo_buffer.set_geometry('buffer_500') #cambiamos la geometría del dataframe para que sea el buffer

restaurantes_geo.drop(['nombre','lat', 'lon', 'dine_in', 'address', 'reservable', 'serves_beer', 'serves_breakfast',
       'serves_brunch', 'serves_dinner', 'serves_lunch',
       'serves_vegetarian_food', 'serves_wine', 'takeout', 'delivery',
       'CODDIS', 'NOMDIS','COD_BAR', 'NOMBRE'], axis=1, inplace=True) #quitamos ciertas variables para que no se repitan

'''
Una vez unficado los 2 dataframes  en la misma métricas vamos a hacer un join espacial donde miramos de la geometria de los restaurantes en que barrio
se encuentra dentro o intersecan.
Quitamos variables que no necesitamos.
'''

result_restaurantes = gpd.sjoin(restaurantes_geo, restaurantes_geo_buffer, how='right', predicate='intersects')
result_restaurantes.drop('index_left', axis=1, inplace=True)
result_restaurantes = result_restaurantes[result_restaurantes['place_id_right'] != result_restaurantes['place_id_left']] #quitamos los id del mismo sitio

''' 
Una vez unidos vamos a agrupar por restaurante y calculamos las medidas antes mencionadas.
- Media de precio de los restaurantes cerca de 500 m.
- Media de rating de los restaurantes cerca de 500 m.
- Media de comentarios de los restaurantes cerca de 500 m.
'''

result = result_restaurantes.groupby(['place_id_right'])[['price_level_left', 'rating_left', 'user_ratings_total_left']].mean().reset_index()
result2 = result_restaurantes.groupby(['place_id_right'])[['place_id_left']].count().reset_index()

result = pd.merge(result, result2, left_on=['place_id_right'], right_on=['place_id_right'])

result.rename({'place_id_right':'place_id',
               'price_level_left':'price_level_mean',
               'rating_left':'rating_mean',
               'user_ratings_total_left':'user_ratings_mean',
               'place_id_left':'num_restaurantes'}, axis=1, inplace=True)

restaurantes = pd.merge(restaurantes, result, how='left', left_on='place_id', right_on='place_id') # Unimos las nuevas columnas en nuestro df de restaurantes

restaurantes.fillna(0, inplace=True) # Rellenamos los null con 0 no tienen restaurantes a 500 m

restaurantes['COD_BAR'] = restaurantes['COD_BAR'].astype('int') # Quitamos los ceros antes de los codigos de barrio
restaurantes['COD_BAR'] = restaurantes['COD_BAR'].astype('str')

#----------------------------------------------------------------------------------------------------------------------------------

##########################################
#####          KPI BARRIOS           #####
##########################################

''' 
Lectura de datos de kpi de barrios Madrid.
'''

kpi = pd.read_csv('../data/raw/kpi_barrios_madrid.csv')

kpi['cod_barrio'] = kpi['cod_barrio'].astype('int') # Quitamos los ceros antes de los codigos de barrio
kpi['cod_barrio'] = kpi['cod_barrio'].astype('str')

''' 
Al ser una fuente de España los decimales se marcan con , y tenemos que reemplazarlos con .
'''

kpi['valor_indicador'] = kpi['valor_indicador'].str.replace(',', '.').astype('float')
kpi['valor_indicador'] = kpi['valor_indicador'].fillna(0) #rellenamos los datos vacios con 0

''' 
Pivotamos la tabla para que los indicadores sea columnas y las rellenamos con 0 los nulls.
'''

kpi = kpi.pivot(index='cod_barrio', columns='indicador_completo', values='valor_indicador').reset_index()
kpi.fillna(0, inplace=True)

''' 
Nos quedamos solo con algunas variables.
'''

kpi = kpi[[ 'cod_barrio', 
            'Año medio de contrucción de inmuebles de uso residencial',
            'Duración media del crédito (meses) en transacción de vivienda',
            'Edad media de la población',
            'Número de locales dados de alta abiertos',
            'Número de locales dados de alta cerrados',
            'Población densidad (hab./Ha.)',
            'Renta disponible media por persona',
            'Tasa de crecimiento demográfico (porcentaje)',
            'Valor catastral medio por inmueble de uso residencial',
            'Tasa absoluta de paro registrado (febrero)',
            'Población de 80 años y más',
            'Primera y segunda nacionalidad extranjera en el Distrito: ITALIA (Población total Distrito)',
            'Primera y segunda nacionalidad extranjera en el Distrito: CHINA (Población total Distrito)']]

restaurantes = pd.merge(restaurantes, kpi, how='left', left_on='COD_BAR', right_on='cod_barrio') # Unimos los indicadores en la base de restaurantes

#----------------------------------------------------------------------------------------------------------------------------------

##########################################
#####          KPI BARRIOS           #####
##########################################

''' 
Lectura de datos de reataurantes de toda España.
'''

street_map = pd.read_csv('../data/raw/sitios_streetmap.csv')

street_map = street_map[(street_map['cocina'].notnull())&(street_map['nombre'].notnull())] # nos quedamos con los registros que la cocina no sea vacia ni su nombre

street_map['cocina'] = [re.sub(r'[^a-zA-Z0-9\s]', ' ', i.lower()) for i in street_map['cocina']] # Hacemos una limpieza previa del tipo de cocina para quitar los caracteres especiales

''' 
Hemos creado una diccionario que mapea el tipo de cocina con una agrupación de cada cocina. 
Este archivo lo cargamos y lo leemos como diccionario.
'''

with open('../data/raw/tipo_cocina.txt', 'r', encoding="utf-8") as file:
    data = file.read()

tipos_cocina = ast.literal_eval(data) #Convertimos a diccionario

''' 
Mapeamos en el data frame de open street map el tipo de cocina y nos quitamos los registro nulos.
'''

street_map['cocina_map'] = street_map['cocina'].map(tipos_cocina)
street_map = street_map[street_map['cocina_map'].notnull()]

''' 
Limpieza de nombre de restaurante para el modelo, quitamos tildes y caracteres especiales.
'''

street_map['nombre'] = street_map['nombre'].apply(u.eliminar_acentos)
street_map['nombre'] = [re.sub(r'[^a-zA-Z0-9\s]', ' ', i.lower()) for i in street_map['nombre']]


#************************************* Modelo **********************************************

X = street_map['nombre']
y = street_map['cocina_map']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

pipe = Pipeline(steps=[("vectorizer", CountVectorizer()),
    ('classifier', MultinomialNB())
])

logistic_params = {
    'vectorizer':[CountVectorizer(stop_words=stopwords.words('spanish'))],
    'classifier': [LogisticRegression(max_iter=10000, solver='liblinear'), LogisticRegression(max_iter=10, solver='liblinear')],
    'classifier__penalty': ['l1', 'l2']
}

naive_param = {
    'vectorizer': [CountVectorizer(stop_words=stopwords.words('spanish'))],
    'classifier': [MultinomialNB()],
    'classifier__alpha': [0.28, 0.30, 0.31],
}

cnaive_param = {
    'vectorizer': [CountVectorizer(stop_words=stopwords.words('spanish'))],
    'classifier': [ComplementNB()],
    'classifier__alpha': [0.1, 0.25, 0.50, 0.75, 0.80, 1],
}

search_space = [
    logistic_params,
    naive_param,
    cnaive_param
]

clf = GridSearchCV(estimator = pipe,
                  param_grid = search_space,
                  cv = 5,
                  n_jobs=-1)

clf.fit(X_train, y_train)

''' 
De todos los modelos que probamos usaremos el modelo de MultinomialNB (bayesianos), que son los que mejor se 
ajustan a modelos de prediccion de textos. 

'''

#print(clf.best_estimator_)
#print(clf.best_score_)
#print(clf.best_params_)

''' 
Evaluamos el modelo en nuestro test.
'''

cocina_mod = clf.best_estimator_
y_pred = cocina_mod.predict(X_test)


#**********************************************************************************************


#----------------------------------------------------------------------------------------------------------------------------------

''' 
Pre procesamos los datos de nombre del restaurantes para poder hacer la predicción  quitando las tildes y caracteres especiales.
'''

restaurantes['nombre'] = restaurantes['nombre'].apply(u.eliminar_acentos)
restaurantes['nombre'] = [re.sub(r'[^a-zA-Z0-9\s]', ' ', i.lower()) for i in restaurantes['nombre']]

restaurantes['tipo_cocina']  = cocina_mod.predict(restaurantes['nombre']) # Creamos columna de tipo de cocina con el modelo

''' 
Proceso para crear las variables dummies de:
- Tipo de cocina.
- Nombre distrito.
- Nombre de barrio.
'''

enc_cocina = OneHotEncoder(handle_unknown='ignore')
tip_coci= enc_cocina.fit_transform(restaurantes[['tipo_cocina']]).toarray()
tip_cocina_dummy = pd.DataFrame(tip_coci, columns=[cat for cat in enc_cocina.categories_[0]])

enc_distrito = OneHotEncoder(handle_unknown='ignore')
tip_distrito= enc_distrito.fit_transform(restaurantes[['NOMDIS']]).toarray()
tip_distrito_dummy = pd.DataFrame(tip_distrito, columns=[cat for cat in enc_distrito.categories_[0]])

enc_barrio = OneHotEncoder(handle_unknown='ignore')
tip_barrio= enc_barrio.fit_transform(restaurantes[['NOMBRE']]).toarray()
tip_barrio_dummy = pd.DataFrame(tip_barrio, columns=[cat for cat in enc_barrio.categories_[0]])

restaurantes = pd.concat([restaurantes, tip_cocina_dummy, tip_distrito_dummy, tip_barrio_dummy], axis=1)

restaurantes.drop(['cod_barrio', 'NOMDIS', 'NOMBRE'], axis=1, inplace=True) # quitamos ciertas columnas que no necesitamos.
''' 
Renombramos las columnas a nombres mas faciles.
'''

restaurantes.rename(columns={
                    'nombre':'nombre_restaurante',
                    'address':'direccion',
                    'CODDIS':'cod_distrito',
                    'COD_BAR':'cod_barrio',
                    'Tasa absoluta de paro registrado (febrero)':'tasa_parados',
                    'Año medio de contrucción de inmuebles de uso residencial':'anio_medio_constr_vivendas',
                    'Duración media del crédito (meses) en transacción de vivienda':'dur_media_credito_viviendas',
                    'Edad media de la población':'edad_media_poblacion',
                    'Número de locales dados de alta abiertos':'num_locales_alta_abiertos',
                    'Número de locales dados de alta cerrados':'num_locales_alta_cerrados',
                    'Población densidad (hab./Ha.)':'poblacion_densidad',
                    'Renta disponible media por persona':'renta_media_persona',
                    'Tasa de crecimiento demográfico (porcentaje)':'pct_crecimiento_demografico',
                    'Valor catastral medio por inmueble de uso residencial':'valor_catast_inmueble_residen',
                    'Población de 80 años y más':'poblacion_80_mas',
                    'Primera y segunda nacionalidad extranjera en el Distrito: ITALIA (Población total Distrito)':'poblacion_italia',
                    'Primera y segunda nacionalidad extranjera en el Distrito: CHINA (Población total Distrito)':'poblacion_china' }, inplace=True)

restaurantes['cod_barrio'] = restaurantes['cod_barrio'].astype(int)
restaurantes['cod_distrito'] = restaurantes['cod_distrito'].astype(int)

''' 
Codificar la variable tipo de cocina en numeros.
'''

encoder = LabelEncoder()
tipo_cocina_encoder = encoder.fit_transform(restaurantes['tipo_cocina'])
restaurantes['tipo_cocina_encoder'] = tipo_cocina_encoder

restaurantes['y'] = restaurantes['rating']*(np.log(restaurantes['user_ratings_total']+1)) # Creamos la variable y o de ponderación.

#----------------------------------------------------------------------------------------------------------------------------------
# Convertimos en Log algunas variables

restaurantes['price_level_mean'] = [np.log1p(i) for i in restaurantes['price_level_mean']]
restaurantes['dur_media_credito_viviendas'] = [np.log1p(i) for i in restaurantes['dur_media_credito_viviendas']]
restaurantes['valor_catast_inmueble_residen'] = [np.log1p(i) for i in restaurantes['valor_catast_inmueble_residen']]
restaurantes['renta_media_persona'] = [np.log1p(i) for i in restaurantes['renta_media_persona']]
restaurantes['tasa_parados'] = [np.log1p(i) for i in restaurantes['tasa_parados']]
restaurantes['poblacion_italia'] = [np.log1p(i) for i in restaurantes['poblacion_italia']]

#----------------------------------------------------------------------------------------------------------------------------------

restaurantes.to_csv('../data/processed/restaurantes.csv', index=False)