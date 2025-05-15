
# Memoria del Proyecto: MadFood

## 📌 Objetivo del Proyecto

El objetivo principal de **MadFood** es proporcionar una herramienta interactiva que ayude a identificar las mejores ubicaciones para abrir un restaurante en Madrid. Se apoya en modelos de machine learning, análisis geoespacial y visualización para ofrecer recomendaciones personalizadas tanto a emprendedores como a empresas del sector restaurantero.

---

## 📁 Estructura del Proyecto

### `01_Fuentes.ipynb` – Recolección de Datos

- Extracción de datos de restaurantes mediante APIs externas.
- Geolocalización, tipos de cocina, valoraciones y más.
- Almacenamiento en formatos estructurados para reutilización.

### `02_LimpiezaEDA.ipynb` – Limpieza y Análisis Exploratorio

- Normalización de texto (tildes, mayúsculas).
- Análisis de correlaciones entre características y popularidad.
- Visualizaciones geoespaciales de restaurantes por distrito.

### `03_Entrenamiento_Evaluacion.ipynb` – Modelado Clásico

- Entrenamiento de modelos como SVM y Random Forest.
- Evaluación de rendimiento con MAE, MSE, etc.
- Almacenamiento de modelos entrenados con `pickle`.

### `03_Entrenamiento_Evaluacion_Red_Neuronal.ipynb` – Modelado Avanzado

- Definición de red neuronal con `Keras`.
- Comparación de métricas con modelos clásicos.
- Visualización de curva de pérdida.

---

## 🌐 Aplicación Streamlit – `app copy.py`

- Diseño visual atractivo con HTML/CSS embebido.
- Dos modos de uso: **Empresas** y **Autónomos**.
- Interfaz con mapas interactivos (`folium` + `geopandas`).
- Uso de modelos entrenados para predicción en tiempo real.
- Navegación controlada por `st.session_state`.

---

## 🧰 Funciones Personalizadas

### Archivo `utils.py`

#### 🔹 `json_to_dataframe(data_entrada)`
Convierte un diccionario JSON (por ejemplo, desde Google Places API) en un `DataFrame` con columnas: `nombre`, `id`, `lat`, `lon`. Facilita el análisis tabular de datos geolocalizados.

#### 🔹 `eliminar_acentos(texto: str)`
Elimina acentos y convierte el texto a minúsculas. Mejora la consistencia del texto al normalizar los datos de entrada para análisis posterior.

---

### Archivo `app.py`

Controla la lógica de interfaz:

- Presenta la pantalla inicial con encabezado e introducción.
- Define botones que activan distintos modos (`empresas` y `autónomos`).
- Llama funciones desde `util_streamlit.py` para mostrar formularios y procesar predicciones:

#### 🔸 `mostrar_formulario_empresas()` *(en `util_streamlit.py`)*
Formulario para negocios establecidos. Permite ingresar información detallada sobre un restaurante y devuelve recomendaciones con mapas.

#### 🔸 `mostrar_formulario_user()` *(en `util_streamlit.py`)*
Diseñado para emprendedores individuales. Ofrece zonas recomendadas con bajo nivel de competencia o alta demanda, con enfoque más simple e intuitivo.

---

## 🧠 Librerías y Tecnologías Usadas

- Python, Pandas, NumPy, Scikit-learn, Keras
- Streamlit (interfaz)
- Folium, GeoPandas (visualización geoespacial)
- Matplotlib, Seaborn (EDA)
- Pickle (modelos persistentes)

---
## ✅ Conclusión

MadFood es un ejemplo completo de cómo aplicar ciencia de datos y machine learning para resolver un problema del mundo real: encontrar la mejor ubicación para un restaurante. La arquitectura modular del proyecto, sus visualizaciones claras y su interfaz intuitiva lo convierten en una herramienta potente y fácil de usar.
