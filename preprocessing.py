import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn.preprocessing as skp

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


pd.options.display.max_columns = None


'''
    Duplica el preprocesamiento explicado en el tp a un dataSet
'''
def preprocesamiento(df, medias_latitud, medias_longitud, moda_dormitorios, ohe_tipo, algo_reduccion):
    completar_latitudes(df, medias_latitud)
    completar_longitudes(df, medias_longitud)
    completar_dormitorios(df, moda_dormitorios)
    df = tratar_tipo(df, ohe_tipo)
    #algo_reduccion.transform(df)
    return df


'''
Completa los nan de longitudes con la media respecto al barrio
La media a usar es la obtenida en el x_train
Nota: Si la media de un barrio es nan, se completara con la media general de x_train
'''
def completar_longitudes(df, medias_longitud):
    valores_place = df.barrio.unique()
    for place in valores_place:
        mask = (df["longitud"].isnull()) & (df["barrio"] == place)
        df.loc[mask, "longitud"] = medias_longitud.get(place, medias_longitud['default'])
        


'''
Completa los nan de latitudes con la media respecto al barrio
La media a usar es la obtenida en el x_train
Nota: Si la media de un barrio es nan, se completara con la media general de x_train
'''
def completar_latitudes(df, medias_latitud):
    valores_place = df.barrio.unique()
    for place in valores_place:
        mask = (df["latitud"].isnull()) & (df["barrio"] == place)
        df.loc[mask, "latitud"] = medias_latitud.get(place, medias_latitud['default']) 


'''
Completa los nan de dormitorios con la moda respecto a los ambientes
La moda a usar es la obtenida en el x_train
Nota: Si la moda de un ambiente es nan, se completara con la moda general de x_train

'''
def completar_dormitorios(df, moda_dormitorios):
    # Completamos los nan de dormitorios con su respectiva moda segun la cantidad de ambientes
    df_na = df[df.dormitorios.isna()]
    unique_rooms = df_na.ambientes.unique().tolist()
    # Le asignamos el valor que corresponda a cada celda de dormitorio que este vacia
    for n in unique_rooms:
        mask = ((df["dormitorios"].isnull())) & (df["ambientes"] == n)
        df.loc[mask, "dormitorios"] = moda_dormitorios.get(n, moda_dormitorios['default'])

'''
Aplica OneHot al feature tipo
'''
def tratar_tipo(df, ohe_tipo):
    f_array = ohe_tipo.transform(df[['tipo']]).toarray()
    f_name_tipo = np.array(ohe_tipo.categories_).ravel()[1:]#Porque elimine el 1ero
    feature_tipo = pd.DataFrame(f_array, columns=f_name_tipo)
    df = pd.concat([df, feature_tipo], axis=1)
    df.drop(columns=["tipo"],inplace=True)
    return df
