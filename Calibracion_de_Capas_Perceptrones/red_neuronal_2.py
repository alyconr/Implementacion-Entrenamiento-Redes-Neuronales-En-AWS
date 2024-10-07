## #######################################################################################################
## 
## @section 1. Librerías
## 
## #######################################################################################################

#Utilitario para extraer parámetros
from argparse import ArgumentParser

#Utilitario para definir la arquitectura de una red neuronal
from keras.models import Sequential

#Utilitario para definir una capa de red neuronal
from keras.layers import Dense

#Utilitario para manipular arrays y matrices
import numpy as np

#Utilitario para manipular el sistema operativo
import os

#Utilitario JSON
import json

#Importamos la implementación del algoritmo de backpropagation
from tensorflow.keras.optimizers import Adam

#Importamos la librería de pandas
import pandas as pd

#Utilitario para dividir los datos en entrenamiento y validación
from sklearn.model_selection import train_test_split

## #######################################################################################################
## 
## @section 2. Arquitectura del modelo
## 
## #######################################################################################################

#Función para construir el modelo
def obtenerModelo(numNeuronas):
  #Instanciamos una arquitectura de red neuronal
  modelo = Sequential()
  
  #Calibración de capa 1
  modelo.add(Dense(numNeuronas, input_shape=(11, ), activation = "relu"))

  #Capa de salida
  modelo.add(Dense(1))
  
  return modelo

## #######################################################################################################
## 
## @section 3. Parámetros
## 
## #######################################################################################################

#Función para obtener los parámetros de entrenamiento
def obtenerArgumentos():
    #Definimos el objeto que nos permite extraer los parámetros de entrenamiento
    parseadorDeParametros = ArgumentParser()
    
    #Parseamos el número de neuronas
    parseadorDeParametros.add_argument("--num-neuronas", type = int, default = 2)
    
    #Parseamos la ruta S3 del directorio de entrenamiento
    #Por defecto es almacenado en una variable del sistema operativo (SM_CHANNEL_TRAINING)
    #La ruta estándar es "/opt/ml/input/data/training", dentro están los archivos S3 de entrenamiento
    parseadorDeParametros.add_argument("--train", type = str, default = os.environ.get("SM_CHANNEL_TRAINING"))
    
    #Parseamos la ruta S3 en donde se almacenará el modelo
    #Por defecto es almacenado en una variable del sistema operativo (SM_MODEL_DIR)
    parseadorDeParametros.add_argument("--sm-model-dir", type = str, default = os.environ.get("SM_MODEL_DIR"))

    #Parseamos la lista de servidores en donde el modelo se está entrenando
    #Por defecto es almacenado en una variable del sistema operativo (SM_HOSTS)
    parseadorDeParametros.add_argument("--hosts", type = list, default = json.loads(os.environ.get("SM_HOSTS")))

    #Parseamos el servidor actual en donde este modelo se está entrenando
    #Por defecto es almacenado en una variable del sistema operativo (SM_CURRENT_HOST)
    parseadorDeParametros.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    #Parseamos el directorio en donde SageMaker almacena metadata de entrenamiento
    #No debemos colocarle ningún valor, sólo crearla, SageMaker la usa internamente
    parseadorDeParametros.add_argument("--model_dir", type = str)
    
    #Parseamos todos los argumentos
    parametros = parseadorDeParametros.parse_args()
    
    #Obtenemos los parámetros
    #IMPORTANTE: No retornamos todos los parámetros
    #IMPORTANTE: Los que no están siendo retornados SageMaker los necesita internamente
    #IMPORTANTE: Si no los colocamos, el modelo no se entrenará
    return parametros.train, parametros.num_neuronas

## #######################################################################################################
## 
## @section 4. Lectura de Datos
## 
## #######################################################################################################

def obtenerDatosEntrenamientoValidacion(rutaDeDatos):
    #Leemos el archivo de dataset
    #dfpDataset = pd.read_csv(rutaDeDatos+"/*", header = None)

    #Obtenemos la lista de archivos desde la ruta
    listaDeArchivos = os.listdir(rutaDeDatos)

    #Lista para almacenar los DataFrames de cada archivo
    dataframes = []
    
    #Itera sobre todos los archivos en el directorio
    for archivo in listaDeArchivos:
        #Nos quedamos sólo con los CSV
        if archivo.endswith('.csv'):
            #Concatenamos el nombre del archivo en la ruta
            rutaCompleta = os.path.join(rutaDeDatos, archivo)

            #Leemos el archivo
            df = pd.read_csv(rutaCompleta, header = None)

            #Lo agregamos a la lista
            dataframes.append(df)
    
    #Concatenamos todos los dataframes
    dfpDataset = pd.concat(dataframes, ignore_index = True)

    #Seleccionamos las columnas features, de la 1 en adelante
    dfpFeatures = dfpDataset[dfpDataset.columns[1:]]

    #Seleccionamos la columna label, la primera
    dfpLabels = dfpDataset[dfpDataset.columns[0]]

    #Usaremos el 80% de los registros para entrenar el modelo y el 20% para validar el modelo
    #Hacemos el corte (split)
    #x_train será el dataframe pandas que tiene los features X de entrenamiento
    #x_test será el dataframe pandas que tiene los features X de validación
    #y_train será el dataframe pandas que tiene los labels Y de entrenamiento
    #y_test será el dataframe pandas que tiene los labels Y de validación
    x_train, x_test, y_train, y_test = train_test_split(dfpFeatures, dfpLabels, test_size=0.2)

    #Retornamos los datos
    return x_train, x_test, y_train, y_test

## #######################################################################################################
## 
## @section 5. Entrenamiento
## 
## #######################################################################################################

#Iniciamos el proceso
if __name__ == '__main__':
    #Obtenemos los parámetros de entrenamiento
    rutaDeDatos, num_neuronas = obtenerArgumentos()

    #Obtenemos la arquitectura del modelo
    modelo = obtenerModelo(num_neuronas)

    #Obtenemos los datos de entrenamiento y validación
    x_train, x_test, y_train, y_test = obtenerDatosEntrenamientoValidacion(rutaDeDatos)

    #Compilamos el modelo
    #Indicamos el ratio de aprendizaje recibido
    modelo.compile(Adam(learning_rate = 0.06263820444233216), "mean_squared_error")

    #Entrenamos el modelo
    modelo.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size = 88)
