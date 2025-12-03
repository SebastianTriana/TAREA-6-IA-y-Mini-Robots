# TAREA 6 -- Inteligencia Artificial y Mini Robots

### Sebastián Triana -- Juan Diego Camacho

### Universidad Nacional de Colombia -- 2025-2

Este repositorio contiene el desarrollo completo de los **4 puntos**
solicitados en la **Tarea #6** del curso *IA y Mini Robots*.\
Los ejercicios abarcan regresión mediante redes neuronales,
clasificación supervisada y análisis de modelos utilizando el dataset
**Iris**.

------------------------------------------------------------------------

# Punto 1 --- Multiplicación de matrices 2×2 usando *MLPRegressor*

En este punto se genera un dataset sintético con **50.000** ejemplos de
multiplicación de matrices 2×2:

\[ C = A `\times `{=tex}B,`\quad `{=tex}A,B,C
`\in `{=tex}`\mathbb{R}`{=tex}\^{2 `\times 2`{=tex}} \]

Se entrena un **MLPRegressor** con:

-   Dos capas ocultas: **128** y **64** neuronas\
-   Activación ReLU\
-   Normalización de entrada y salida\
-   Evaluación con **MAE**\
-   Comparación con 10 ejemplos nuevos

**Archivo:** `punto1.py`

------------------------------------------------------------------------

# Punto 2 --- Clasificación Iris con SVC + GridSearchCV

Se entrena un modelo **SVC** para clasificar las 3 clases del dataset
Iris.\
Incluye:

-   Estandarización\
-   GridSearchCV con hiperparámetros (C, `\gamma`{=tex}, kernel)\
-   Reporte de *best params*\
-   Accuracy y *classification report*

**Archivo:** `punto2.py`

------------------------------------------------------------------------

# Punto 3 --- Clasificación Iris con KNN + Validación cruzada

En este punto se experimenta con **KNeighborsClassifier** probando
valores de (k):

-   Estandarización\
-   Validación cruzada 5-fold\
-   Selección del mejor (k)\
-   Evaluación final en test

**Archivo:** `punto3.py`

------------------------------------------------------------------------

# Punto 4 --- Clasificación Iris con Árbol de Decisión

Se entrena un **DecisionTreeClassifier** con profundidad máxima 3.\
Incluye:

-   Partición de dataset\
-   Accuracy en test\
-   Impresión de reglas del árbol usando `export_text`

**Archivo:** `punto4.py`

------------------------------------------------------------------------

# Requisitos

Instalar dependencias ejecutando:

``` bash
pip install numpy scikit-learn
```

------------------------------------------------------------------------

# Ejecución

Cada punto puede ejecutarse con:

``` bash
python punto1.py
python punto2.py
python punto3.py
python punto4.py
```

------------------------------------------------------------------------

# Autores

-   Sebastián Triana
-   Juan Diego Camacho

Universidad Nacional de Colombia\
Facultad de Ingeniería -- 2025-2
