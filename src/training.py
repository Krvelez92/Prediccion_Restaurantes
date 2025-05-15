##########################################
#####          LIBRERIAS             #####
##########################################

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
#----------------------------------------------------------------------------------------------------------------------------------

''' 
Lectura de datos de restaurantes de Madrid.
'''
restaurantes = pd.read_csv('../data/processed/restaurantes.csv')

restaurantes.drop(['nombre_restaurante', 'place_id', 
                   'direccion', 'tipo_cocina',
                   'rating', 'user_ratings_total'
                   ], inplace=True, axis=1)

#----------------------------------------------------------------------------------------------------------------------------------
##########################################
#####           MODELO               #####
##########################################

def modelo(X, y):
    ''' 
    Funcion para ejecutar el modelo entrenado final.

    Inputs:
        X:DataFrame
        y:DataFrame
    
    Output:
        model:sklearn.pipeline.Pipeline
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # Extraer df de train y test.

    train.to_csv('../data/processed/train/train.csv', index=False)
    test.to_csv('../data/processed/test/test.csv', index=False)

    print('Guardado Train y Test.')

    # Creamos el pipeline

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVR())])

    SVR_param = {
        'scaler': [MinMaxScaler(), StandardScaler()],
        'classifier': [SVR()],
        'classifier__kernel': ['linear', 'poly', 'rbf'],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__degree': [2, 3, 4, 5],
        'classifier__C':[0.5, 1, 10, 50, 100],
        'classifier__max_iter': [1000000]
    }


    search_space = [
        SVR_param
    ]

    clf = GridSearchCV(estimator = pipe,
                    param_grid = search_space,
                    scoring='neg_mean_absolute_error',
                    cv = 10,
                    n_jobs=-1)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model

#----------------------------------------------------------------------------------------------------------------------------------
X = restaurantes.drop(['y', 'cod_distrito', 'tipo_cocina_encoder', 'cod_barrio'], axis=1)
y = restaurantes['y']

print('Iniciando Entrenamiento')

best_model = modelo(X, y)

filename = '../models/final_model.pkl'

print('Guardando Modelo')

with open(filename, 'wb') as archivo_salida:
    pickle.dump(best_model, archivo_salida)

print('Finalizando Entrenamiento')