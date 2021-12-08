import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    dataset['gender'] = dataset['gender'].astype('category')
    dataset['gender'] = dataset['gender'].cat.codes
    dataset['class'] = dataset['class'].astype('category')
    dataset['class'] = dataset['class'].cat.codes
    return dataset

dataset = load_dataset("../dataset/bodyPerformance.csv")

titles = dataset.columns.values
values = dataset.values

x = values[:,:-1]
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
y = values[:,-1]
title_x = titles
title_x = np.delete(title_x, 11, 0)
title_y = titles[11]

n_clases = dataset["class"].nunique()

param_svm = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
param_knn = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
param_decisiontree = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
param_randomforest = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
param_logireg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

param_grid = [param_svm, param_knn, param_decisiontree, param_randomforest, param_logireg]

models = [svm.SVC(probability=True), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "KNN", "Decision Tree", "Random Forest", "Logistic Regression"]

for i,model in enumerate(models):
    '''Busqueda exhaustiva de los mejores parametros'''
    print("BUSQUEDA EXHAUSTIVA DE PARAMETROS")
    grid = GridSearchCV(model, param_grid[i], verbose=3, n_jobs=-1)
    grid.fit(x,y)
    print("Els millors parametres: ",grid.best_params_)
    print("El millor score: ", grid.best_score_)
    print("")
