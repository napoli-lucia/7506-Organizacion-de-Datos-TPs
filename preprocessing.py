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
import geopandas as geopd
import rtree
from geopandas.tools import sjoin


warnings.simplefilter(action="ignore", category=FutureWarning)



pd.options.display.max_columns = None


'''
    Duplica el preprocesamiento explicado en el tp a un dataSet
'''
def preprocesamiento(df, feature_eliminar, medias_latitud, medias_longitud, moda_dormitorios, ohe_tipo, df_comunas):
    completar_latitudes(df, medias_latitud)
    completar_longitudes(df, medias_longitud)
    completar_dormitorios(df, moda_dormitorios)
    df = tratar_tipo(df, ohe_tipo)
    df =crear_comunas(df, df_comunas)
    eliminar_features(df, feature_eliminar)
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


def crear_comunas(df, df_comunas):
    dp_comunas = geopd.read_file("./comunas.csv")
    df_comunas = dp_comunas.copy()
    
    df_comunas=df_comunas.filter(['COMUNAS','geometry'])
    df_comunas.rename(columns={'COMUNAS':'comuna'},inplace=True)
    
    # Casteamos las comunas a enteros para mayor comodidad
    df_comunas['comuna']=df_comunas.comuna.astype(float)
    df_comunas['comuna']=df_comunas.comuna.astype(int)
    
    # Unimos los dataframes asignando la comuna que le corresponde a cada punto segun su coordenada
    tam_inicial=df.shape[0]
    df_comunas.set_crs('EPSG:4326', inplace=True)
    df = df.sjoin(df_comunas, how="inner")
    df.drop(columns='index_right', inplace=True)

    #Ya no necesitare el feature geometry
    df.drop(columns=["geometry"],inplace=True)
    df = df.sort_index()
    return df


def eliminar_features(df, feature_eliminar):
    df.drop(columns=feature_eliminar,inplace=True)