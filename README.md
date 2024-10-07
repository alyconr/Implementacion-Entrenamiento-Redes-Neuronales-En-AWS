# IMPLEMENTACION Y ENTRENAMIENTO DE REDES NEURONALES EN AWS  CASO HIPOTETICO DE SEGUROS MEDICOS
## Índice
1. [Introducción](#introducción)
2. [Configuración del Entorno](#configuración-del-entorno)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Implementación del Modelo](#implementación-del-modelo)
   4.1 [Calibración de Hiperparámetros](#calibración-de-hiperparámetros)
   4.2 [Script de Entrenamiento](#script-de-entrenamiento)
   4.3 [Calibración de Perceptrones](#calibración-de-perceptrones)
   4.4 [Calibración de Función de Activación](#calibración-de-función-de-activación)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Evaluación y Selección del Mejor Modelo](#evaluación-y-selección-del-mejor-modelo)
7. [Conclusiones y Trabajo Futuro](#conclusiones-y-trabajo-futuro)

## 1. Introducción

Este repositorio contiene la implementación, construcción y entrenamiento de un modelo de redes neuronales utilizando Amazon SageMaker. El proyecto se enfoca en la predicción de costos de seguros utilizando técnicas de aprendizaje profundo y calibración de hiperparámetros.

## 2. Configuración del Entorno

El proyecto utiliza las siguientes bibliotecas y servicios:

- Amazon SageMaker
- Boto3
- TensorFlow 2.6
- Keras
- NumPy
- Pandas
- Scikit-learn

Para configurar el entorno, asegúrese de tener instaladas estas dependencias y configurado el acceso a AWS.

## 3. Estructura del Proyecto

El proyecto está estructurado de la siguiente manera:

- `source/`: Directorio que contiene los scripts de entrenamiento
  - `red_neuronal_1.py`: Script inicial para el entrenamiento de la red neuronal
  - `red_neuronal_2.py`: Script mejorado con calibración de perceptrones
- Notebooks de Jupyter para la configuración y ejecución de experimentos

## 4. Implementación del Modelo

### 4.1 Calibración de Hiperparámetros

Se implementó un proceso de calibración de hiperparámetros utilizando el `HyperparameterTuner` de SageMaker. Los hiperparámetros ajustados incluyen:

- Learning rate: Rango continuo de 0.0001 a 0.1
- Batch size: Rango entero de 32 a 128
- Número de neuronas: Valores categóricos [2, 4, 8, 16, 32, 64, 128]

### 4.2 Script de Entrenamiento

El script de entrenamiento (`red_neuronal_1.py`) incluye:

- Definición de la arquitectura del modelo usando Keras
- Parseo de argumentos para hiperparámetros
- Carga y preprocesamiento de datos
- Entrenamiento del modelo con validación

### 4.3 Calibración de Perceptrones

En `red_neuronal_2.py`, se implementó la calibración del número de perceptrones en la capa oculta, permitiendo experimentar con diferentes arquitecturas de red.

### 4.4 Calibración de Función de Activación

Aunque no se muestra explícitamente en los scripts proporcionados, se recomienda experimentar con diferentes funciones de activación como ReLU, tanh, o sigmoid para optimizar el rendimiento del modelo.

## 5. Entrenamiento del Modelo

El entrenamiento se realiza utilizando SageMaker, con las siguientes características:

- Uso de instancias `ml.m5.large`
- Distribución de datos "FullyReplicated"
- 100 épocas de entrenamiento
- Validación cruzada con 20% de los datos

## 6. Evaluación y Selección del Mejor Modelo

La evaluación y selección del mejor modelo se realiza mediante:

- Métrica de evaluación: validation loss (mean squared error)
- Objetivo: Minimizar la pérdida de validación
- Selección automática del mejor modelo basado en el rendimiento

## 7. Conclusiones y Trabajo Futuro

Este proyecto demuestra la implementación exitosa de un modelo de red neuronal para la predicción de costos de seguros. Futuras mejoras podrían incluir:

- Experimentación con arquitecturas más complejas
- Incorporación de técnicas de regularización
- Exploración de otros algoritmos de optimización

Para cualquier pregunta o sugerencia, por favor abra un issue en este repositorio.
