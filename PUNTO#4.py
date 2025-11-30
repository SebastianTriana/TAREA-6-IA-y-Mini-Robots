# PUNTO #4: Clasificación del dataset Iris usando DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
dt = DecisionTreeClassifier(max_depth=3, random_state=0)
dt.fit(Xtr, ytr)
print("Test acc:", accuracy_score(yte, dt.predict(Xte)))
# imprimir reglas
r = export_text(dt, feature_names=iris.feature_names)
print("\nReglas del árbol:\n", r)
