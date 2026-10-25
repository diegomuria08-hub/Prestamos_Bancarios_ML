# Predicción de aprobación de préstamos con XGBoost

Proyecto educativo de Machine Learning para explorar el entrenamiento y uso de un clasificador binario. El modelo usa tres variables de entrada: ingresos anuales, score de crédito y edad.

## Estructura

```text
.
├── .gitignore                  # Archivos locales excluidos de Git
├── predecir_prestamo.py         # Interfaz interactiva de predicción
├── proyecto_modelo_banco.ipynb  # Notebook original de entrenamiento en Colab
├── requirements.txt             # Dependencias de Python
└── README.md
```

El script entrena el modelo en memoria con los mismos datos de ejemplo del notebook; no necesita un archivo de modelo serializado. Así puedes revisar y modificar el entrenamiento directamente en `predecir_prestamo.py` o experimentar en el notebook.

## Requisitos

Se recomienda Python 3.9 o posterior. Instala las dependencias desde la terminal de VS Code:

```bash
py -m pip install -r requirements.txt
```

## Entrenamiento del modelo

El notebook `proyecto_modelo_banco.ipynb` contiene el flujo de entrenamiento para Google Colab. El script de terminal también entrena un `XGBClassifier` en memoria al iniciarse, usando el mismo conjunto pequeño de datos simulados. Para guardar un modelo desde el notebook como parte de tus experimentos, puedes usar:

```python
import pandas as pd
from xgboost import XGBClassifier

datos_banco = {
    "Ingresos": [45000, 32000, 85000, 120000, 25000, 55000, 95000, 30000, 110000, 40000],
    "Score_Credito": [650, 580, 710, 790, 500, 680, 740, 600, 810, 620],
    "Edad": [34, 28, 45, 50, 22, 39, 41, 26, 48, 31],
    "Pago_Prestamo": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
}

datos = pd.DataFrame(datos_banco)
columnas = ["Ingresos", "Score_Credito", "Edad"]
X_entreno = datos[columnas]
y_entreno = datos["Pago_Prestamo"]

modelo = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=2,
    random_state=42,
    eval_metric="logloss",
)
modelo.fit(X_entreno, y_entreno)
probabilidades = modelo.predict_proba(X_entreno)
```

Las columnas de entrada deben conservar los nombres y el orden `Ingresos`, `Score_Credito` y `Edad`.

## Ejecutar una predicción

Desde la terminal, en la carpeta del proyecto:

```bash
py -u predecir_prestamo.py
```

El programa entrena el modelo una sola vez al iniciar y luego solicita ingresos anuales en USD, score de crédito y edad para cada cliente. Convierte los datos a `float` y los envía al modelo en un DataFrame con las columnas `Ingresos`, `Score_Credito` y `Edad`. Si se introduce texto, solicita el valor de nuevo. Tras cada resultado puedes analizar otro cliente respondiendo `S` o terminar con `N` o Enter. También muestra las probabilidades que el clasificador asigna a pago y no pago.

## Alcance y limitaciones

Este ejemplo sirve para aprender el flujo básico de clasificación e inferencia con XGBoost. El conjunto de entrenamiento tiene solo diez filas simuladas y no incluye una evaluación de calibración; sus probabilidades no representan la confianza de un banco ni son confiables para decisiones financieras reales. Para un uso serio se necesitarían datos representativos, evaluación con datos separados, calibración de probabilidades, análisis de sesgos y validación del modelo.
