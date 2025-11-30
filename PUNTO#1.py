# PUNTO #1: Multiplicación de matrices 2x2 usando MLPRegressor
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 1) Generar dataset
def gen_dataset(n_samples=50000, low=-20, high=20, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 8), dtype=float)  # A(4) + B(4)
    Y = np.zeros((n_samples, 4), dtype=float)  # A*B flattened (2x2 -> 4)
    for i in range(n_samples):
        A = rng.integers(low, high+1, size=(2,2))
        B = rng.integers(low, high+1, size=(2,2))
        C = A @ B
        X[i, :4] = A.flatten()
        X[i, 4:] = B.flatten()
        Y[i, :] = C.flatten()
    return X, Y

print("Generando dataset ...")
X, Y = gen_dataset(50000)

# 2) Partición
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.15, random_state=1)

# 3) Escalado 
scX = StandardScaler()
scY = StandardScaler()
X_tr_s = scX.fit_transform(X_tr)
X_te_s = scX.transform(X_te)

Y_tr_s = scY.fit_transform(Y_tr)
Y_te_s = scY.transform(Y_te)

# 4) Modelo: MLPRegressor (regresión multivariable)
model = MLPRegressor(hidden_layer_sizes=(128,64), activation='relu',
                     solver='adam', max_iter=200, random_state=1, verbose=True)

print("Entrenando MLPRegressor...")
model.fit(X_tr_s, Y_tr_s)

# 5) Evaluación en test set
Y_pred_s = model.predict(X_te_s)
Y_pred = scY.inverse_transform(Y_pred_s)

mae = mean_absolute_error(Y_te, Y_pred)
print(f"\nMAE en test (elemento a elemento): {mae:.4f}")

# 6) 10 ejemplos
rng = np.random.default_rng(123)
print("\nComparación en 10 ejemplos nuevos:")
for i in range(10):
    A = rng.integers(-20, 21, size=(2,2))
    B = rng.integers(-20, 21, size=(2,2))
    X_sample = np.hstack([A.flatten(), B.flatten()]).reshape(1,-1)
    Xs = scX.transform(X_sample)
    pred_s = model.predict(Xs)
    pred = scY.inverse_transform(pred_s).reshape(2,2).round(3)
    true = (A @ B).astype(float)
    print(f"\nEj {i+1}:")
    print("A =\n", A)
    print("B =\n", B)
    print("Producto (analítico):\n", true)
    print("Predicción del modelo:\n", pred)
    print("Error absoluto medio (muestra):", np.mean(np.abs(true - pred)).round(4))
