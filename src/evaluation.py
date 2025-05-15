##########################################
#####          LIBRERIAS             #####
##########################################

import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#----------------------------------------------------------------------------------------------------------------------------------
# Lectura Modelo
with open('../models/final_model.pkl', 'rb') as archivo_entrada:
    modelo_importado = pickle.load(archivo_entrada)

# lectura de Test
restaurantes_test = pd.read_csv('../data/processed/test/test.csv', index_col=0)

# Separa X y Y
X = restaurantes_test.drop(['y'], axis=1)
y = restaurantes_test['y']

#----------------------------------------------------------------------------------------------------------------------------------

predictions = modelo_importado.predict(X)

print("MAE test", mean_absolute_error(y, predictions))
print("MAPE test", mean_absolute_percentage_error(y, predictions))
print("MSE test", mean_squared_error(y, predictions))
print("RMSE test", mean_squared_error(y, predictions)**(1/2))
print("R2 score", r2_score(y, predictions))