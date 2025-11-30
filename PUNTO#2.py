# PUNTO #2: Clasificación del dataset Iris usando SVC con GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
sc = StandardScaler().fit(X_train)
X_train_s = sc.transform(X_train)
X_test_s = sc.transform(X_test)

# grid busqueda simple
param_grid = {'C':[0.1,1,10], 'gamma':['scale','auto'], 'kernel':['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_s, y_train)
print("Best params:", grid.best_params_)

y_pred = grid.predict(X_test_s)
print("Acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
