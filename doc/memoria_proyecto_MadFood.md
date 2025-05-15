
# Memoria del Proyecto: MadFood

## 📌 Descripción General

**MadFood** es una aplicación interactiva construida con **Streamlit** que ayuda a descubrir la mejor ubicación para abrir un restaurante en Madrid, basándose en datos de popularidad, geolocalización y predicciones de modelos de machine learning. El proyecto combina análisis exploratorio, modelado predictivo y visualización interactiva.

---

## 📁 Estructura del Proyecto

- `01_Fuentes.ipynb`: Obtención de datos desde fuentes externas.
- `02_LimpiezaEDA.ipynb`: Limpieza de datos y análisis exploratorio.
- `03_Entrenamiento_Evaluacion.ipynb`: Entrenamiento de modelos predictivos.
- `03_Entrenamiento_Evaluacion_Red_Neuronal.ipynb`: Versión alternativa usando redes neuronales.
- `app copy.py`: Aplicación Streamlit que actúa como interfaz de usuario final.
- `utils.py`: Funciones auxiliares como transformación de JSON y limpieza de texto.

---

## 🔍 Metodología

### 1. Recolección de Datos (`01_Fuentes.ipynb`)
Se accedió a APIs (como Google Places u otras) para obtener información geolocalizada de restaurantes en Madrid.

### 2. Limpieza y EDA (`02_LimpiezaEDA.ipynb`)
- Eliminación de nulos y duplicados
- Estandarización de nombres y tipos de restaurantes
- Análisis de correlaciones y visualización geográfica

### 3. Modelado (`03_Entrenamiento_Evaluacion.ipynb`)
- Se utilizaron modelos como SVM y Random Forest para predecir popularidad o probabilidad de éxito de un restaurante según su ubicación y características.
- Se evaluó el rendimiento mediante métricas como F1-score y accuracy.
- También se entrenó un modelo de red neuronal en un notebook aparte.

---

## 🧠 Arquitectura de la App (`app copy.py`)

La aplicación se ejecuta con Streamlit e incluye dos modos de uso:

- **Empresas**: Enfocado a negocios consolidados
- **Autónomos**: Recomendaciones personalizadas para emprendedores

Incluye funciones como:

- Formularios para ingreso de parámetros del usuario
- Visualización geográfica con Folium
- Resultados basados en modelos entrenados previamente

El archivo `utils.py` incluye funciones auxiliares como `json_to_dataframe` para convertir entradas JSON a DataFrames y `eliminar_acentos` para normalizar texto.

---

## 🛠 Tecnologías Usadas

- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Streamlit
- Folium y GeoPandas (mapas interactivos)
- Jupyter Notebooks
- Pickle (para guardar modelos entrenados)

---

## 📦 Requisitos

```bash
pip install -r requirements.txt
```

Contenido sugerido para `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
folium
geopandas
shapely
requests
```
---

## ✨ Conclusión

MadFood es una solución interactiva basada en ciencia de datos para apoyar decisiones estratégicas en el sector gastronómico. Desde la recolección de datos hasta la visualización interactiva, integra múltiples capas de análisis para facilitar la apertura de nuevos restaurantes con mayor probabilidad de éxito.
