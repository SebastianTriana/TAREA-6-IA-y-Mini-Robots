# TAREA 5 – Inteligencia Artificial y Mini Robots  
### Sebastián Triana – Juan Diego Camacho  
### Universidad Nacional de Colombia – 2025-2

Este repositorio contiene el desarrollo completo de los 3 puntos solicitados en la **Tarea #5** del curso *IA y Mini Robots*.  
Se implementan modelos de redes neuronales, clasificación con TensorFlow (Fashion-MNIST) y un ejercicio con dataset externo aplicando una red neuronal y análisis de pesos aprendidos.

---

## Estructura del repositorio

TAREA-5-IA-y-Mini-Robots/
│
├── punto1_NAND_XOR.py
├── punto2_fashion_mnist.py
├── punto3_dataset_externo.py
│
├── resultados/
│ ├── nand_predicciones.txt
│ ├── xor_predicciones.txt
│ ├── fashion_mnist_metricas.txt
│ ├── dataset_externo_pesos.txt
│
└── README.md

yaml
Copy code

---

# Punto 1 — Redes neuronales para NAND y XOR

En este punto se entrenaron **dos redes neuronales** con TensorFlow/Keras, ambas con **2 capas ocultas**, para aprender a aproximar el comportamiento de las compuertas:

- **NAND**
- **XOR**

Cada red se entrenó con backpropagation y luego se imprimieron:

- Predicciones sobre las cuatro combinaciones posibles.
- Los **pesos aprendidos** en cada capa.

**Archivos relevantes:**

- `punto1_NAND_XOR.py`
- `resultados/nand_predicciones.txt`
- `resultados/xor_predicciones.txt`

---

# Punto 2 — Clasificación con Fashion-MNIST usando TensorFlow

Se descargó el dataset **Fashion-MNIST** y se entrenó un modelo simple de clasificación usando:

- Capa `Flatten`
- Capa oculta densa
- Capa de salida softmax (10 clases)

Se presentan:

- Accuracy en entrenamiento y prueba  
- Matriz de confusión  
- Ejemplos correctamente e incorrectamente clasificados  

**Archivo relevante:**  
`punto2_fashion_mnist.py`

---

# Punto 3 — Dataset externo y red neuronal personalizada

Se eligió el dataset **Iris** de scikit-learn.  
El pipeline incluyó:

1. Visualización de *features* y *clases*.  
2. One-hot encoding del rótulo (corregido usando `OneHotEncoder(sparse_output=False)`).  
3. Entrenamiento de una red neuronal fully-connected.  
4. Métricas de desempeño.  
5. **Extracción e interpretación de los pesos aprendidos**.

**Archivo relevante:**  
`punto3_dataset_externo.py`

---

# ⚙️ Requisitos

Instalar dependencias:

```bash
pip install numpy tensorflow scikit-learn matplotlib
```
#Cómo ejecutar cada punto

##Punto 1: NAND y XOR
```bash
python punto1_NAND_XOR.py
```
##Punto 2: Fashion MNIST
```bash
python punto2_fashion_mnist.py
```
Punto 3: Dataset externo (Iris)
```bash
python punto3_dataset_externo.py
```
#Autores
Sebastián Triana
Juan Diego Camacho

Universidad Nacional de Colombia
Facultad de Ingeniería – 2025-2
