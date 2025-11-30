# PUNTO #3: Clasificación del dataset Iris usando KNeighborsClassifier con validación cruzada
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data; y = iris.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xte_s = sc.transform(Xte)

# probar k's
best_k, best_score = 1, 0
for k in [1,3,5,7,9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    s = cross_val_score(knn, Xtr_s, ytr, cv=5).mean()
    if s > best_score:
        best_score, best_k = s, k
print("Mejor k:", best_k, "score CV:", best_score)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(Xtr_s, ytr)
print("Test acc:", accuracy_score(yte, knn.predict(Xte_s)))
